from functools import partial
from typing import NamedTuple, Tuple
import os

import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
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


# Simple 5x5 pixel font for text rendering
PIXEL_FONT = {
    'P': [
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,0,0,0,0],
    ],
    'R': [
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,1,0,0],
        [1,0,0,1,0],
    ],
    'E': [
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,1,1,0,0],
        [1,0,0,0,0],
        [1,1,1,1,1],
    ],
    'S': [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,1,1,1,0],
    ],
    'A': [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
    ],
    'N': [
        [1,0,0,0,1],
        [1,1,0,0,1],
        [1,0,1,0,1],
        [1,0,0,1,1],
        [1,0,0,0,1],
    ],
    'Y': [
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,1,0,1,0],
    ],
    'T': [
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
    ],
    'H': [
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
    ],
    'I': [
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
    ],
    'G': [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,1,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
    ],
    'M': [
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,1,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
    ],
    'O': [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
    ],
    'V': [
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,0,1,0,0],
    ],
    'W': [
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,1,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1],
    ],
    ' ': [
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
}

def render_text(canvas, text, start_x, start_y, color, font_size=1):
    """Render text using pixel font on canvas"""
    x_offset = 0
    for char in text.upper():
        if char in PIXEL_FONT:
            pattern = PIXEL_FONT[char]
            for row_idx, row in enumerate(pattern):
                for col_idx, pixel in enumerate(row):
                    if pixel == 1:
                        px = start_x + x_offset + col_idx * font_size
                        py = start_y + row_idx * font_size
                        # Draw pixel
                        for fy in range(font_size):
                            for fx in range(font_size):
                                canvas = canvas.at[py + fy, px + fx, :].set(color)
            x_offset += 6 * font_size  # 5 pixels + 1 space
        else:
            x_offset += 6 * font_size  # Space for unknown character
    return canvas

def get_text_width(text, font_size=1):
    """Calculate the width of text in pixels"""
    width = 0
    for char in text.upper():
        if char in PIXEL_FONT:
            width += 6 * font_size  # 5 pixels + 1 space
        else:
            width += 6 * font_size  # Space for unknown character
    return width

def render_centered_text(canvas, text, start_y, color, font_size=1, screen_width=160):
    """Render text centered horizontally on screen"""
    text_width = get_text_width(text, font_size)
    start_x = (screen_width - text_width) // 2
    return render_text(canvas, text, start_x, start_y, color, font_size)


class MsPacmanConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    cell_size: int = 4
    grid_width: int = 40
    grid_height: int = 44
    num_ghosts: int = 2
    pellet_reward: int = 1
    power_pellet_reward: int = 5
    ghost_reward: int = 10
    collision_penalty: int = -5
    frightened_duration: int = 60
    initial_lives: int = 3
    max_steps: int = 2000
    ghost_spawn_delay: int = 60
    ghost_move_period: int = 6  # ghosts move every N frames
    player_move_period: int = 3  # pacman moves every N frames
    ghost_move_period: int = 2  # ghosts move every N frames
    pellet_mask: Tuple[Tuple[int, ...], ...] = None  # Will be set below
    background_color: Tuple[int, int, int] = (0, 20, 100)
    wall_color: Tuple[int, int, int] = (200, 50, 50)
    blocked_color: Tuple[int, int, int] = (0, 20, 100)
    pellet_color: Tuple[int, int, int] = (200, 50, 50)
    power_pellet_color: Tuple[int, int, int] = (255, 255, 255)
    button_color: Tuple[int, int, int] = (255, 105, 180)
    pacman_color: Tuple[int, int, int] = (255, 255, 0)
    button_power_duration: int = 40
    ghost_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),
        (255, 184, 222),
        (0, 255, 255),
        (255, 128, 0),
    )
    
    
    
    maze_layout: Tuple[str, ...] = (
        "1111111111111111111111111111111111111111",
        "1000000000100000000000000000010000000001",
        "10222222201022222222222222220102222G2201",
        "1020000020102000000000000002010200000201",
        "1020111020102011111111111102010201110201",
        "1020111020102011111111111102010201110201",
        "1020111020102011111111111102010201110201",
        "1020000020002000000000000002000200000201",
        "1022222222222222222222222222222222222201",
        "1000200020000020000000000200000200020001",
        "1110201020111020111111110201110201020111",
        "1110201020111020111111110201110201020111",
        "1110201020111020111111110201110201020111",
        "0000201020000020000000000200000201020000",
        "2222201022222222222222222222222201022222",
        "0000201000000020000000000200000001020000",
        "1110201111111020111111110201111111020111",
        "1110201111111020100000010201111111020111",
        "1110201111111020100000010201111111020111",
        "0010200000000020100000010200000000020100",
        "0010222222222220100000010222222222220100",
        "0010200000000020100000010200000000020100",
        "1110201111111020100000010201111111020111",
        "1110201111111020111111110201111111020111",
        "1110201111111020111111110201111111020111",
        "0000201000000020000000000200000001020000",
        "2222201022222222222P22222222222201022222",
        "0000201020002000200000020002000201020000",
        "1110201020102010201111020102010201020111",
        "1110201020102010201111020102010201020111",
        "1110201020102010201111020102010201020111",
        "1000200020102010200000020102010200020001",
        "1022222220102010222222220102010222222201",
        "1020000020102010200000020102010200000201",
        "1020111020102000201111020002010201110201",
        "1020111020102222201111022222010201110201",
        "1020111020100000201111020000010201110201",
        "1020111020111110201111020111110201110201",
        "1020111020111110201111020111110201110201",
        "1020111020111110201111020111110201110201",
        "1020000020000000200000020000000200000201",
        "1022222222222222222222222222222222222201",
        "1000000000000000000000000000000000000001",
        "1111111111111111111111111111111111111111",
    )

# Define pellet mask for evenly spaced dots in corridors
pellet_mask: Tuple[Tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
)


# Create default constants instance with pellet mask
DEFAULT_MSPACMAN_CONSTANTS = MsPacmanConstants(
    screen_width=160,
    screen_height=210,
    cell_size=4,
    grid_width=40,
    grid_height=44,
    num_ghosts=2,
    pellet_reward=1,
    power_pellet_reward=5,
    ghost_reward=10,
    collision_penalty=-5,
    frightened_duration=60,
    initial_lives=3,
    max_steps=2000,
    ghost_spawn_delay=60,
    ghost_move_period=6,
    player_move_period=3,
    background_color=(0, 20, 100),
    wall_color=(200, 50, 50),
    blocked_color=(0, 20, 100),
    pellet_color=(200, 50, 50),
    power_pellet_color=(255, 255, 255),
    button_color=(255, 105, 180),
    pacman_color=(255, 255, 0),
    button_power_duration=40,
    ghost_colors=(
        (255, 0, 0),
        (255, 184, 255),
        (0, 255, 255),
        (255, 184, 82),
    ),
    maze_layout=(
        "1111111111111111111111111111111111111111",
        "1000000000100000000000000000010000000001",
        "10222222201022222222222222220102222G2201",
        "1020000020102000000000000002010200000201",
        "1020111020102011111111111102010201110201",
        "1020111020102011111111111102010201110201",
        "1020111020102011111111111102010201110201",
        "1020000020002000000000000002000200000201",
        "1022222222222222222222222222222222222201",
        "1000200020000020000000000200000200020001",
        "1110201020111020111111110201110201020111",
        "1110201020111020111111110201110201020111",
        "1110201020111020111111110201110201020111",
        "0000201020000020000000000200000201020000",
        "2222201022222222222222222222222201022222",
        "0000201000000020000000000200000001020000",
        "1110201111111020111111110201111111020111",
        "1110201111111020100000010201111111020111",
        "1110201111111020100000010201111111020111",
        "0010200000000020100000010200000000020100",
        "0010222222222220100000010222222222220100",
        "0010200000000020100000010200000000020100",
        "1110201111111020100000010201111111020111",
        "1110201111111020111111110201111111020111",
        "1110201111111020111111110201111111020111",
        "0000201000000020000000000200000001020000",
        "2222201022222222222P22222222222201022222",
        "0000201020002000200000020002000201020000",
        "1110201020102010201111020102010201020111",
        "1110201020102010201111020102010201020111",
        "1110201020102010201111020102010201020111",
        "1000200020102010200000020102010200020001",
        "1022222220102010222222220102010222222201",
        "1020000020102010200000020102010200000201",
        "1020111020102000201111020002010201110201",
        "1020111020102222201111022222010201110201",
        "1020111020100000201111020000010201110201",
        "1020111020111110201111020111110201110201",
        "1020111020111110201111020111110201110201",
        "1020111020111110201111020111110201110201",
        "1020000020000000200000020000000200000201",
        "1022222222222222222222222222222222222201",
        "1000000000000000000000000000000000000001",
        "1111111111111111111111111111111111111111",
    ),
    pellet_mask=pellet_mask
)


class MsPacmanState(NamedTuple):
    pacman_x: chex.Array
    pacman_y: chex.Array
    direction: chex.Array
    ghost_positions: chex.Array
    pellets: chex.Array
    power_timer: chex.Array
    score: chex.Array
    time: chex.Array
    lives: chex.Array
    pellets_remaining: chex.Array
    game_over: chex.Array
    game_phase: chex.Array  # 0=start_screen, 1=playing, 2=game_over
    game_over_waiting: chex.Array  # 1=waiting for input to restart


def _parse_layout(layout: Tuple[str, ...], expected_ghosts: int):
    wall_rows = []
    pellet_rows = []
    button_rows = []
    pacman_spawn = None
    ghost_positions: list[Tuple[int, int]] = []

    for y, row in enumerate(layout):
        wall_row = []
        pellet_row = []
        button_row = []
        for x, char in enumerate(row):
            if char == "#":
                wall_row.append(1)
                pellet_row.append(0)
                button_row.append(0)
            elif char == "1":
                wall_row.append(1)
                pellet_row.append(0)
                button_row.append(0)
            elif char == "2":
                # Traversable path (pellets controlled by separate mask)
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
            elif char == "0":
                # Visually same as 2 but blocked for movement (use pellet=-1 to signal visual)
                wall_row.append(1)
                pellet_row.append(-1)
                button_row.append(0)
            elif char == ".":
                wall_row.append(0)
                pellet_row.append(1)
                button_row.append(0)
            elif char == "o":
                wall_row.append(0)
                pellet_row.append(2)
                button_row.append(0)
            elif char == "B":
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(1)
            elif char == "P":
                pacman_spawn = (x, y)
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
            elif char == "G":
                ghost_positions.append((x, y))
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
            else:
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
        wall_rows.append(wall_row)
        pellet_rows.append(pellet_row)
        button_rows.append(button_row)

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
        jnp.array(button_rows, dtype=jnp.int32),
    )


class JaxMsPacman(JaxEnvironment[MsPacmanState, MsPacmanObservation, MsPacmanInfo, MsPacmanConstants]):
    def __init__(self, consts: MsPacmanConstants = None, reward_funcs: list[callable] = None):
        consts = consts or DEFAULT_MSPACMAN_CONSTANTS
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        (
            self.wall_grid,
            self.initial_pellets,
            self.pacman_spawn,
            self.ghost_spawn_positions,
            self.initial_pellet_count,
            self.button_grid,
        ) = _parse_layout(self.consts.maze_layout, self.consts.num_ghosts)

        # Preserve visual template (contains -1 markers for blocked-but-path-colored tiles)
        pellet_template = self.initial_pellets

        grid_h, grid_w = self.wall_grid.shape

        # Generate pellets using maze layout '2' markers (already defined positions)
        grid_h, grid_w = self.wall_grid.shape
        pellets = jnp.zeros((grid_h, grid_w), dtype=jnp.int32)
        
        # Place pellets where maze_layout has '2' markers
        for y, row in enumerate(self.consts.maze_layout):
            for x, char in enumerate(row):
                if char == '2':  # Pellet position in maze layout
                    pellets = pellets.at[y, x].set(1)
                elif char == 'G':  # Power pellet position
                    pellets = pellets.at[y, x].set(2)

        # Thin out provided mask to every other tile as well
        #if self.consts.pellet_mask:
            #checker = (jnp.add.outer(jnp.arange(grid_h), jnp.arange(grid_w)) % 2) == 0
            #pellets = jnp.where(checker, pellets, 0)

        self.initial_pellets = pellets
        self.initial_pellet_count = int(jnp.sum(pellets > 0))

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

        self.state = self.reset()[1]  # This should start in start screen
        self.renderer = MsPacmanRenderer(
            self.consts,
            self.wall_grid,
            pellet_template,
            self.button_grid,
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[MsPacmanObservation, MsPacmanState]:
        state = MsPacmanState(
            pacman_x=self.pacman_spawn[0],
            pacman_y=self.pacman_spawn[1],
            direction=jnp.array([-1, 0], dtype=jnp.int32),  # Face left initially
            ghost_positions=self.ghost_spawn_positions,
            pellets=self.initial_pellets,
            power_timer=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.initial_lives, dtype=jnp.int32),
            pellets_remaining=self.initial_pellet_count,
            game_over=jnp.array(False, dtype=jnp.bool_),
            game_phase=jnp.array(0, dtype=jnp.int32),  # Start in start screen
            game_over_waiting=jnp.array(0, dtype=jnp.int32),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MsPacmanState, action: int) -> Tuple[MsPacmanObservation, MsPacmanState, float, bool, MsPacmanInfo]:
        action = jnp.asarray(action, dtype=jnp.int32)
        
        # Handle different game phases
        def handle_start_screen(_):
            # Only transition to playing on non-NOOP action (action != 0)
            def start_game(_):
                new_state = state._replace(game_phase=jnp.array(1, dtype=jnp.int32))
                return self._get_observation(new_state), new_state, jnp.float32(0.0), False, self._get_info(new_state)
            
            def stay_in_start(_):
                return self._get_observation(state), state, jnp.float32(0.0), False, self._get_info(state)
            
            return jax.lax.cond(
                action != 0,  # Only start on non-NOOP action
                start_game,
                stay_in_start,
                None
            )
        
        def handle_game_over(_):
            # If we just entered game over, set waiting flag
            def set_waiting_flag(_):
                new_state = state._replace(game_over_waiting=jnp.array(1, dtype=jnp.int32))
                return self._get_observation(new_state), new_state, jnp.float32(0.0), False, self._get_info(new_state)
            
            # Already waiting, now reset to start screen on non-NOOP action
            def reset_to_start(_):
                new_state = MsPacmanState(
                    pacman_x=self.pacman_spawn[0],
                    pacman_y=self.pacman_spawn[1],
                    direction=jnp.array([-1, 0], dtype=jnp.int32),
                    ghost_positions=self.ghost_spawn_positions,
                    pellets=self.initial_pellets,
                    power_timer=jnp.array(0, dtype=jnp.int32),
                    score=jnp.array(0, dtype=jnp.int32),
                    time=jnp.array(0, dtype=jnp.int32),
                    lives=jnp.array(self.consts.initial_lives, dtype=jnp.int32),
                    pellets_remaining=self.initial_pellet_count,
                    game_over=jnp.array(False, dtype=jnp.bool_),
                    game_phase=jnp.array(0, dtype=jnp.int32),  # Back to start screen
                    game_over_waiting=jnp.array(0, dtype=jnp.int32),
                )
                return self._get_observation(new_state), new_state, jnp.float32(0.0), False, self._get_info(new_state)
            
            def stay_in_game_over(_):
                return self._get_observation(state), state, jnp.float32(0.0), False, self._get_info(state)
            
            return jax.lax.cond(
                state.game_over_waiting == 0,
                set_waiting_flag,
                lambda _: jax.lax.cond(
                    action != 0,  # Only reset on non-NOOP action
                    reset_to_start,
                    stay_in_game_over,
                    None
                ),
                None
            )
        
        def handle_playing():
            # Normal game logic
            return self._step_game(state, action)
        
        # Branch based on game phase
        return jax.lax.cond(
            state.game_phase == 0,
            handle_start_screen,
            lambda _: jax.lax.cond(
                state.game_phase == 2,
                handle_game_over,
                lambda _: handle_playing(),
                None
            ),
            None
        )
    
    def _step_game(self, state: MsPacmanState, action: int) -> Tuple[MsPacmanObservation, MsPacmanState, float, bool, MsPacmanInfo]:
        action_idx = jnp.clip(action, 0, self._action_deltas.shape[0] - 1)
        delta = self._action_deltas[action_idx]

        has_new_direction = jnp.any(delta != 0)
        direction = jnp.where(has_new_direction, delta, state.direction)
        can_move = (state.time % jnp.maximum(self.consts.player_move_period, 1)) == 0

        tentative_x = jnp.mod(state.pacman_x + direction[0], self.consts.grid_width)
        tentative_y = jnp.clip(state.pacman_y + direction[1], 0, self.consts.grid_height - 1)

        hit_wall = self.wall_grid[tentative_y, tentative_x] == 1
        pacman_x = jnp.where(hit_wall, state.pacman_x, tentative_x)
        pacman_y = jnp.where(hit_wall, state.pacman_y, tentative_y)
        zero_dir = jnp.array([0, 0], dtype=jnp.int32)
        direction = jnp.where(hit_wall, zero_dir, direction)
        pacman_x = jnp.where(can_move, pacman_x, state.pacman_x)
        pacman_y = jnp.where(can_move, pacman_y, state.pacman_y)

        pellet_value = state.pellets[pacman_y, pacman_x]
        ate_pellet = pellet_value > 0
        ate_power = pellet_value == 2

        pellets = state.pellets.at[pacman_y, pacman_x].set(jnp.where(ate_pellet, 0, pellet_value))

        pellet_reward = jnp.where(
            ate_power,
            self.consts.power_pellet_reward,
            jnp.where(ate_pellet, self.consts.pellet_reward, 0),
        )
        score = state.score + pellet_reward
        pellets_remaining = jnp.maximum(
            state.pellets_remaining - ate_pellet.astype(jnp.int32),
            0,
        )

        decayed_timer = jnp.maximum(state.power_timer - 1, 0)
        power_timer = jnp.where(
            ate_power,
            jnp.array(self.consts.frightened_duration, dtype=jnp.int32),
            decayed_timer,
        )
        on_button = self.button_grid[pacman_y, pacman_x] == 1
        power_timer = jnp.where(
            on_button,
            jnp.array(self.consts.button_power_duration, dtype=jnp.int32),
            power_timer,
        )
        frightened = power_timer > 0

        ghost_positions = self._move_ghosts(state.ghost_positions, pacman_x, pacman_y, frightened, state.time)

        ghost_overlap = jnp.logical_and(
            ghost_positions[:, 0] == pacman_x,
            ghost_positions[:, 1] == pacman_y,
        )
        ghosts_eaten = jnp.logical_and(ghost_overlap, frightened)
        ghost_bonus = jnp.sum(ghosts_eaten.astype(jnp.int32)) * self.consts.ghost_reward
        score = score + ghost_bonus

        ghost_positions = jnp.where(
            jnp.broadcast_to(ghosts_eaten[:, None], ghost_positions.shape),
            self.ghost_spawn_positions,
            ghost_positions,
        )

        any_collision = jnp.any(ghost_overlap)
        hit_without_power = jnp.logical_and(any_collision, jnp.logical_not(frightened))

        score = score + jnp.where(hit_without_power, self.consts.collision_penalty, 0)
        lives = state.lives - hit_without_power.astype(jnp.int32)

        pacman_x = jax.lax.select(hit_without_power, self.pacman_spawn[0], pacman_x)
        pacman_y = jax.lax.select(hit_without_power, self.pacman_spawn[1], pacman_y)
        ghost_positions = jax.lax.select(hit_without_power, self.ghost_spawn_positions, ghost_positions)
        power_timer = jax.lax.select(hit_without_power, jnp.array(0, dtype=jnp.int32), power_timer)
        left_dir = jnp.array([-1, 0], dtype=jnp.int32)  # Face left when hit
        direction = jax.lax.select(hit_without_power, left_dir, direction)

        time = state.time + 1

        game_over = jnp.logical_or(
            state.game_over,
            jnp.logical_or(
                lives <= 0,
                jnp.logical_or(pellets_remaining <= 0, time >= self.consts.max_steps),
            ),
        )

        # Transition to game over phase when lives run out
        game_phase = jnp.where(lives <= 0, jnp.array(2, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32))
        game_over_waiting = jnp.where(lives <= 0, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

        new_state = MsPacmanState(
            pacman_x=pacman_x,
            pacman_y=pacman_y,
            direction=direction,
            ghost_positions=ghost_positions,
            pellets=pellets,
            power_timer=power_timer,
            score=score,
            time=time,
            lives=lives,
            pellets_remaining=pellets_remaining,
            game_over=game_over,
            game_phase=game_phase,
            game_over_waiting=game_over_waiting,
        )

        done = self._get_done(new_state)
        env_reward = jnp.float32(self._get_reward(state, new_state))
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

        dx_sign = jnp.where(frightened, -dx_sign, dx_sign)
        dy_sign = jnp.where(frightened, -dy_sign, dy_sign)

        prefer_x = jnp.abs(dx) >= jnp.abs(dy)
        primary_dx = jnp.where(prefer_x, dx_sign, 0)
        primary_dy = jnp.where(prefer_x, 0, dy_sign)
        secondary_dx = jnp.where(prefer_x, 0, dx_sign)
        secondary_dy = jnp.where(prefer_x, dy_sign, 0)

        cand1_x = jnp.mod(gx + primary_dx, self.consts.grid_width)
        cand1_y = jnp.clip(gy + primary_dy, 0, self.consts.grid_height - 1)
        blocked1 = self.wall_grid[cand1_y, cand1_x] == 1

        cand2_x = jnp.mod(gx + secondary_dx, self.consts.grid_width)
        cand2_y = jnp.clip(gy + secondary_dy, 0, self.consts.grid_height - 1)
        blocked2 = self.wall_grid[cand2_y, cand2_x] == 1

        use_second = jnp.logical_and(blocked1, jnp.logical_not(blocked2))
        stay = jnp.logical_and(blocked1, blocked2)

        new_x = jnp.where(stay, gx, jnp.where(use_second, cand2_x, cand1_x))
        new_y = jnp.where(stay, gy, jnp.where(use_second, cand2_y, cand1_y))
        new_x = jnp.where(can_move, new_x, gx)
        new_y = jnp.where(can_move, new_y, gy)

        # Add more randomness so ghosts are less perfect: 70% chance to take a random step (including staying put).
        key = jax.random.PRNGKey(time)
        keys = jax.random.split(key, self.consts.num_ghosts + 1)
        noise_mask = jax.random.bernoulli(keys[0], p=0.7, shape=(self.consts.num_ghosts,))
        rand_keys = keys[1:]
        rand_idx = jax.vmap(lambda k: jax.random.randint(k, (), 0, 5))(rand_keys)
        dir_table = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=jnp.int32)
        rand_delta = dir_table[rand_idx]
        rand_x = jnp.mod(gx + rand_delta[:, 0], self.consts.grid_width)
        rand_y = jnp.clip(gy + rand_delta[:, 1], 0, self.consts.grid_height - 1)
        rand_blocked = self.wall_grid[rand_y, rand_x] == 1
        rand_x = jnp.where(rand_blocked, gx, rand_x)
        rand_y = jnp.where(rand_blocked, gy, rand_y)

        final_x = jnp.where(noise_mask, rand_x, new_x)
        final_y = jnp.where(noise_mask, rand_y, new_y)

        # Respect spawn delays: inactive ghosts stay at their spawn positions.
        spawn_x = self.ghost_spawn_positions[:, 0]
        spawn_y = self.ghost_spawn_positions[:, 1]
        final_x = jnp.where(active, final_x, spawn_x)
        final_y = jnp.where(active, final_y, spawn_y)

        return jnp.stack([final_x, final_y], axis=1).astype(jnp.int32)

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
    def __init__(self, consts: MsPacmanConstants = None, wall_grid: jnp.ndarray = None, pellet_template: jnp.ndarray = None, button_grid: jnp.ndarray = None,):
        super().__init__()
        self.consts = consts or DEFAULT_MSPACMAN_CONSTANTS
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        if wall_grid is None:
            wall_grid = jnp.zeros((self.consts.grid_height, self.consts.grid_width), dtype=jnp.int32)
        self.wall_grid = jnp.asarray(wall_grid, dtype=jnp.int32)
        if pellet_template is None:
            pellet_template = jnp.zeros_like(self.wall_grid)
        self.pellet_template = jnp.asarray(pellet_template, dtype=jnp.int32)
        if button_grid is None:
            button_grid = jnp.zeros_like(self.wall_grid)
        self.button_grid = jnp.asarray(button_grid, dtype=jnp.int32)

        grid_h, grid_w = self.wall_grid.shape
        grid_px_w = grid_w * self.consts.cell_size
        grid_px_h = grid_h * self.consts.cell_size
        self.offset_x = max((self.consts.screen_width - grid_px_w) // 2, 0)
        self.offset_y = max((self.consts.screen_height - grid_px_h) // 2, 0)

        self.background_color = jnp.asarray(self.consts.background_color, dtype=jnp.uint8)
        self.wall_color = jnp.asarray(self.consts.wall_color, dtype=jnp.uint8)
        self.blocked_color = jnp.asarray(self.consts.blocked_color, dtype=jnp.uint8)
        self.pellet_color = jnp.asarray(self.consts.pellet_color, dtype=jnp.uint8)
        self.power_pellet_color = jnp.asarray(self.consts.power_pellet_color, dtype=jnp.uint8)
        self.button_color = jnp.asarray(self.consts.button_color, dtype=jnp.uint8)
        self.pacman_color = jnp.asarray(self.consts.pacman_color, dtype=jnp.uint8)
        self.ghost_colors = jnp.asarray(self.consts.ghost_colors, dtype=jnp.uint8)
        self.score_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        self.score_bar_color = jnp.array([0, 0, 0], dtype=jnp.uint8)
        # 3x5 pixel font for digits 0-9
        self.digit_patterns = jnp.array([
            [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],  # 0
            [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],  # 1
            [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],  # 2
            [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]],  # 3
            [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],  # 4
            [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],  # 5
            [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],  # 6
            [[1,1,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0]],  # 7
            [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],  # 8
            [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],  # 9
        ], dtype=jnp.uint8)
        self.background_image = None
        self.pacman_sprites = self._load_pacman_sprites()
        self.ghost_sprites = self._load_ghost_sprites()
        self._base_grid_pixels = self._create_base_grid()
        self._base_canvas = self._create_base_canvas(self._base_grid_pixels)

    def _load_pacman_sprites(self) -> jnp.ndarray | None:
        """Load Ms. Pac-Man sprites from the sprites directory."""
        try:
            sprite_dir = os.path.join(os.path.dirname(__file__), "sprites", "mspacman")
            sprites = []
            for i in range(4):
                path = os.path.join(sprite_dir, f"pacman_{i}.npy")
                sprite_rgba = np.load(path)
                sprites.append(jnp.asarray(sprite_rgba, dtype=jnp.uint8))
            return jnp.stack(sprites)  # Shape: (4, height, width, channels)
        except Exception:
            return None

    def _load_ghost_sprites(self) -> jnp.ndarray | None:
        """Load ghost sprites from the sprites directory."""
        try:
            sprite_dir = os.path.join(os.path.dirname(__file__), "sprites", "mspacman")
            # Order: blinky, pinky, inky, sue, blue, white
            ghost_types = ["blinky", "pinky", "inky", "sue", "blue", "white"]
            sprites = []
            for ghost_type in ghost_types:
                path = os.path.join(sprite_dir, f"ghost_{ghost_type}.npy")
                sprite_rgba = np.load(path)
                sprites.append(jnp.asarray(sprite_rgba, dtype=jnp.uint8))
            return jnp.stack(sprites)  # Shape: (6, height, width, channels)
        except Exception:
            return None

    def _rotate_sprite(self, sprite: jnp.ndarray, direction: int) -> jnp.ndarray:
        """Rotate sprite based on direction: 0=left, 1=up, 2=right, 3=down"""
        # JAX-compatible rotation using conditional logic
        def rotate_left(): return sprite  # Base sprite faces left
        def rotate_up(): return jnp.rot90(sprite, k=3)  # 270° clockwise
        def rotate_right(): return jnp.fliplr(sprite)  # Flip horizontally instead of rotate
        def rotate_down(): return jnp.rot90(sprite, k=1)  # 90° clockwise
        
        return jax.lax.cond(
            direction == 0, rotate_left,
            lambda: jax.lax.cond(
                direction == 1, rotate_up,
                lambda: jax.lax.cond(
                    direction == 2, rotate_right,
                    rotate_down
                )
            )
        )

    def _create_base_grid(self) -> jnp.ndarray:
        """Pre-draw the static maze (background + walls) once."""
        cell = self.consts.cell_size
        wall_h, wall_w = self.wall_grid.shape
        base_shape = (wall_h, wall_w, 3)
        soft_mask = jnp.logical_and(self.wall_grid == 1, self.pellet_template == -1)
        wall_mask = jnp.logical_and(self.wall_grid == 1, jnp.logical_not(soft_mask))[..., None]
        blocked_mask = soft_mask[..., None]  # 0-tiles marked with -1 render as path color
        background_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.background_color
        wall_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.wall_color
        blocked_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.blocked_color
        grid = jnp.where(wall_mask, wall_layer, background_layer)
        grid = jnp.where(blocked_mask, blocked_layer, grid)
        grid_pixels = jnp.repeat(grid, cell, axis=0)
        grid_pixels = jnp.repeat(grid_pixels, cell, axis=1)
        return grid_pixels

    def _create_base_canvas(self, grid_pixels: jnp.ndarray) -> jnp.ndarray:
        """Place the pre-drawn maze onto a background-colored canvas."""
        canvas_h = max(self.consts.screen_height, self.offset_y + grid_pixels.shape[0])
        canvas_w = max(self.consts.screen_width, self.offset_x + grid_pixels.shape[1])
        canvas = jnp.ones((canvas_h, canvas_w, 3), dtype=jnp.uint8) * self.background_color
        if self.background_image is not None:
            bg = self.background_image
            h = min(bg.shape[0], canvas.shape[0])
            w = min(bg.shape[1], canvas.shape[1])
            alpha = bg[:h, :w, 3:4].astype(jnp.uint16)
            fg_rgb = bg[:h, :w, :3].astype(jnp.uint16)
            bg_rgb = canvas[:h, :w, :].astype(jnp.uint16)
            blended = ((fg_rgb * alpha) + (bg_rgb * (255 - alpha))) // 255
            canvas = canvas.at[:h, :w, :].set(blended.astype(jnp.uint8))
        return canvas.at[
            self.offset_y:self.offset_y + grid_pixels.shape[0],
            self.offset_x:self.offset_x + grid_pixels.shape[1],
        ].set(grid_pixels)

    def _alpha_blend(self, canvas: jnp.ndarray, patch: jnp.ndarray, top: int, left: int) -> jnp.ndarray:
        """Alpha blend an RGBA patch onto canvas at (top, left)."""
        h, w = patch.shape[0], patch.shape[1]
        region = jax.lax.dynamic_slice(canvas, (top, left, 0), (h, w, 3))
        alpha = patch[:, :, 3:4].astype(jnp.uint16)
        fg_rgb = patch[:, :, :3].astype(jnp.uint16)
        bg_rgb = region.astype(jnp.uint16)
        blended = ((fg_rgb * alpha) + (bg_rgb * (255 - alpha))) // 255
        return jax.lax.dynamic_update_slice(canvas, blended.astype(jnp.uint8), (top, left, 0))

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: MsPacmanState) -> jnp.ndarray:
        # Always render the game first
        canvas = self._render_game(state)
        
        # Add overlay text based on game phase
        def add_start_overlay(canvas):
            # Add "PRESS START" text centered using pixel font (shorter text, smaller font)
            text_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White
            canvas = render_centered_text(canvas, "PRESS START", 100, text_color, font_size=2, screen_width=self.consts.screen_width)
            return canvas
        
        def add_game_over_overlay(canvas):
            # Add "GAME OVER" text centered using pixel font (smaller font to fit)
            over_color = jnp.array([255, 0, 0], dtype=jnp.uint8)  # Red
            canvas = render_centered_text(canvas, "GAME OVER", 80, over_color, font_size=2, screen_width=self.consts.screen_width)
            return canvas
        
        # Apply overlay based on game phase
        canvas = jax.lax.cond(
            state.game_phase == 0,
            lambda: add_start_overlay(canvas),
            lambda: jax.lax.cond(
                state.game_phase == 2,
                lambda: add_game_over_overlay(canvas),
                lambda: canvas  # No overlay during gameplay
            )
        )
        
        return canvas
    
    def _render_game(self, state: MsPacmanState) -> jnp.ndarray:
        cell = self.consts.cell_size

        pellet_tiles = (state.pellets == 1)[..., None]
        power_tiles = (state.pellets == 2)[..., None]
        button_tiles = (self.button_grid == 1)[..., None]

        grid_pixels = self._base_grid_pixels
        pellet_pixels = jnp.repeat(pellet_tiles, cell, axis=0)
        pellet_pixels = jnp.repeat(pellet_pixels, cell, axis=1)
        power_pixels = jnp.repeat(power_tiles, cell, axis=0)
        power_pixels = jnp.repeat(power_pixels, cell, axis=1)
        button_pixels = jnp.repeat(button_tiles, cell, axis=0)
        button_pixels = jnp.repeat(button_pixels, cell, axis=1)

        # Create proper index arrays for the repeated grid
        grid_h, grid_w = grid_pixels.shape[:2]
        row_idx = jnp.arange(grid_h) % cell
        col_idx = jnp.arange(grid_w) % cell
        row_idx = row_idx[:, None]
        col_idx = col_idx[None, :]

        pellet_half = jnp.maximum(cell // 6, 1)
        power_half = jnp.maximum(cell // 4, 2)

        # Single dot pellet in center of cell (for 4x4 cell, center is at 1,1)
        center_row = (cell - 1) // 2  # For cell=4, this gives 1
        center_col = (cell - 1) // 2  # For cell=4, this gives 1
        pellet_center = jnp.logical_and(
            row_idx == center_row,
            col_idx == center_col,
        )[..., None]
        power_center = jnp.logical_and(
            jnp.abs(row_idx - cell // 2) < power_half,
            jnp.abs(col_idx - cell // 2) < power_half,
        )[..., None]

        pellet_pixels = jnp.logical_and(pellet_pixels, pellet_center)
        power_pixels = jnp.logical_and(power_pixels, power_center)
        button_center = jnp.logical_and(
            jnp.abs(row_idx - cell // 2) < power_half,
            jnp.logical_or(
                jnp.abs(row_idx - cell // 4) < power_half // 2,
                jnp.abs(col_idx - cell // 4) < power_half // 2,
            ),
        )[..., None]
        button_pixels = jnp.logical_and(button_pixels, button_center)

        pellet_layer_px = jnp.ones_like(grid_pixels) * self.pellet_color
        power_layer_px = jnp.ones_like(grid_pixels) * self.power_pellet_color
        button_layer_px = jnp.ones_like(grid_pixels) * self.button_color
        grid_pixels = jnp.where(pellet_pixels, pellet_layer_px, grid_pixels)
        grid_pixels = jnp.where(power_pixels, power_layer_px, grid_pixels)
        grid_pixels = jnp.where(button_pixels, button_layer_px, grid_pixels)

        canvas = self._base_canvas
        canvas = canvas.at[
            self.offset_y:self.offset_y + grid_pixels.shape[0],
            self.offset_x:self.offset_x + grid_pixels.shape[1],
        ].set(grid_pixels)

        # ---- Draw score bar at top ----
        def _score_to_digits(score_val, max_digits=6):
            def body(i, carry):
                val, out = carry
                digit = val % 10
                out = out.at[max_digits - 1 - i].set(digit)
                val = val // 10
                return (val, out)
            _, digits = jax.lax.fori_loop(
                0, max_digits, body, (jnp.maximum(score_val, 0), jnp.zeros((max_digits,), dtype=jnp.int32))
            )
            return digits

        def _draw_digits(img, digits, scale=2, pad=2):
            pat = self.digit_patterns
            dh, dw = pat.shape[1] * scale, pat.shape[2] * scale
            bar_h = dh + pad * 2
            bar_w = min(img.shape[1], pad * 2 + digits.shape[0] * (dw + 1))
            # Draw bar
            img = img.at[0:bar_h, 0:bar_w, :].set(self.score_bar_color)

            def place_digit(carry, idx):
                img = carry
                digit = digits[idx]
                pattern = pat[digit]
                scaled = jnp.kron(pattern, jnp.ones((scale, scale), dtype=jnp.uint8))
                block = scaled[:, :, None] * self.score_color
                x = pad + idx * (dw + 1)
                y = pad
                img = jax.lax.dynamic_update_slice(img, block, (y, x, 0))
                return img, None

            img, _ = jax.lax.scan(place_digit, img, jnp.arange(digits.shape[0]))
            return img

        digits = _score_to_digits(state.score, max_digits=6)
        canvas = _draw_digits(canvas, digits, scale=2, pad=2)

        pac_px = self.offset_x + state.pacman_x * cell
        pac_py = self.offset_y + state.pacman_y * cell
        
        # Draw Ms. Pac-Man with animated and directional sprite
        if self.pacman_sprites is not None:
            # Determine direction: 0=left, 1=up, 2=right, 3=down
            dir_x, dir_y = state.direction[0], state.direction[1]
            
            # Map direction to rotation (base sprite faces left)
            direction_idx = jnp.where(dir_x < 0, 0,  # left
                            jnp.where(dir_y < 0, 1,  # up
                            jnp.where(dir_x > 0, 2,  # right
                            3)))  # down (default)
            
            # Animation frame based on time
            anim_frame = (state.time // 4) % 4
            
            # Select sprite based on direction and animation
            # We'll rotate sprites based on direction
            base_sprite = self.pacman_sprites[anim_frame]
            rotated_sprite = self._rotate_sprite(base_sprite, direction_idx)
            
            # Center sprite in cell
            sx = (cell - rotated_sprite.shape[1]) // 2
            sy = (cell - rotated_sprite.shape[0]) // 2
            canvas = self._alpha_blend(canvas, rotated_sprite, pac_py + sy, pac_px + sx)
        else:
            pac_block = jnp.ones((cell, cell, 3), dtype=jnp.uint8) * self.pacman_color
            canvas = jax.lax.dynamic_update_slice(canvas, pac_block, (pac_py, pac_px, 0))

        ghost_positions = state.ghost_positions
        # Draw ghosts with proper sprites
        if self.ghost_sprites is not None:
            def draw_ghost_sprite(img, inputs):
                idx, pos = inputs
                # Use frightened state or normal ghost colors - use JAX-compatible indexing
                frightened = state.power_timer > 0
                transition_time = state.power_timer > 30
                
                # JAX-compatible sprite selection
                normal_idx = idx % 4  # 0-3 for normal ghosts
                blue_idx = jnp.array(4, dtype=jnp.int32)  # blue sprite
                white_idx = jnp.array(5, dtype=jnp.int32)  # white sprite
                
                # Choose sprite based on frightened state
                frightened_sprite = jnp.where(transition_time, blue_idx, white_idx)
                sprite_idx = jnp.where(frightened, frightened_sprite, normal_idx)
                
                ghost_sprite = self.ghost_sprites[sprite_idx]  # Use JAX array indexing
                gx = self.offset_x + pos[0] * cell
                gy = self.offset_y + pos[1] * cell
                # Center sprite in cell
                sx = (cell - ghost_sprite.shape[1]) // 2
                sy = (cell - ghost_sprite.shape[0]) // 2
                return self._alpha_blend(img, ghost_sprite, gy + sy, gx + sx), None
            
            canvas, _ = jax.lax.scan(
                draw_ghost_sprite,
                canvas,
                (jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), ghost_positions),
            )
        else:
            # Fallback to colored blocks if sprites not available
            color_indices = jnp.mod(jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), self.ghost_colors.shape[0])
            ghost_palette = self.ghost_colors[color_indices]
            ghost_size = cell * 3
            ghost_blocks = jnp.ones((self.consts.num_ghosts, ghost_size, ghost_size, 3), dtype=jnp.uint8) * ghost_palette[:, None, None, :]

            def draw_ghost(img, inputs):
                idx, pos = inputs
                block = ghost_blocks[idx]
                # Center larger ghost block on tile
                gx = self.offset_x + pos[0] * cell - (ghost_size - cell) // 2
                gy = self.offset_y + pos[1] * cell - (ghost_size - cell) // 2
                max_x = img.shape[1] - ghost_size
                max_y = img.shape[0] - ghost_size
                gx = jnp.clip(gx, 0, max_x)
                gy = jnp.clip(gy, 0, max_y)
                img = jax.lax.dynamic_update_slice(img, block, (gy, gx, 0))
                return img, None

            canvas, _ = jax.lax.scan(
                draw_ghost,
                canvas,
                (jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), ghost_positions),
            )

        # Draw lives display near score (top-left area)
        def draw_life_icon(canvas, idx):
            # Position lives icons at top-left, below score area
            life_x = 10 + idx * 20
            life_y = 30
            # Draw small Pac-Man icon for each life (only if idx < remaining lives)
            life_color = jnp.where(idx < state.lives, jnp.array(self.consts.pacman_color, dtype=jnp.uint8), jnp.array([0, 0, 0], dtype=jnp.uint8))
            life_block = jnp.ones((8, 8, 3), dtype=jnp.uint8) * life_color
            canvas = jax.lax.dynamic_update_slice(canvas, life_block, (life_y, life_x, 0))
            return canvas, None

        # Draw up to 3 life icons (fixed maximum for JAX compatibility)
        canvas, _ = jax.lax.scan(
            draw_life_icon,
            canvas,
            jnp.arange(3, dtype=jnp.int32),  # Fixed maximum of 3 lives
        )

        return canvas






















import time
import pygame
import jax
import jax.random as jr
from jaxatari.games.jax_mspacman import JaxMsPacman
from jaxatari.environment import JAXAtariAction as Action

UPSCALE = 4
FPS = 8

def main():
    pygame.init()
    env = JaxMsPacman()
    render = jax.jit(env.render)
    step = jax.jit(env.step)

    obs, state = env.reset()
    frame = render(state)
    h, w = frame.shape[:2]
    window = pygame.display.set_mode((w * UPSCALE, h * UPSCALE))
    clock = pygame.time.Clock()

    rng = jr.PRNGKey(0)
    running = True
    action = Action.NOOP

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = Action.UP
        elif keys[pygame.K_DOWN]:
            action = Action.DOWN
        elif keys[pygame.K_LEFT]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Action.RIGHT
        else:
            action = Action.NOOP

        obs, state, reward, done, info = step(state, action)
        if done:
            obs, state = env.reset()

        frame = render(state)
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (w * UPSCALE, h * UPSCALE))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
