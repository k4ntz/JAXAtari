# Hacking tips: 
# - use scripts\amidar_maze_generator.py to change the maze
# - you can change the number of enemies for the starting levels, the maximal number of enemies and at which level this switch happens in the constants. 
# - changing the enemy_types each level works by just overwriting the get_enemy_types function
# - In general, there are a lot of constants which can be changed in order to change the behavior 

# remaining Inacuracies:
# - the bottom path is the same as any other path, in ALE it's thinner and the sprites are further up on the path

from functools import partial
import os
from typing import NamedTuple, Tuple
import chex
import jax
import jax.numpy as jnp
from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.games.amidar_mazes import original as chosen_maze # change this to change the maze

# Functions to precompute some constants so they only need to be calculated once

@partial(jax.jit, static_argnames=['WIDTH', 'HEIGHT', 'PATH_THICKNESS_HORIZONTAL', 'PATH_THICKNESS_VERTICAL'])
def generate_path_mask(WIDTH, HEIGHT, PATH_THICKNESS_HORIZONTAL, PATH_THICKNESS_VERTICAL, horizontal_edges, vertical_edges, corners, horizontal_cond, vertical_cond, corner_cond):
    """Generates a mask for the path edges. Conditions are used so this function can also be used to render only the walked on paths."""
    # Create an empty mask 
    mask = jnp.zeros((WIDTH, HEIGHT), dtype=jnp.int32)
    rendering_mask = jnp.zeros((WIDTH, HEIGHT), dtype=jnp.int32)

    def interpolate_horizontal_line(edge, condition, num_points=WIDTH): # Could use less points if performance is a problem, but this accounts for all valid maze changes

        def calculate():
            start, end = edge

            # for the computational path mask
            xs = jnp.linspace(start[0], end[0], num_points).astype(jnp.int32)
            ys = jnp.full(xs.shape, start[1], dtype=jnp.int32)  # y-coordinate is constant for horizontal lines
            coords = jnp.stack([xs, ys], axis=-1)  # shape (num_points, 2)

            # for the rendering path mask
            xs_rendering = jnp.linspace(start[0]+PATH_THICKNESS_VERTICAL, end[0]-1, num_points).astype(jnp.int32) # only render the edges at the parts between the corners, the corners are rendered seperately
            y_offsets = jnp.arange(PATH_THICKNESS_HORIZONTAL)  # offsets for the vertical expansion
            ys_rendering = start[1] + y_offsets[:, None]

            # make sure that the xs and ys are the same shape
            shape = jnp.broadcast_shapes(ys_rendering.shape, xs_rendering.shape)
            xs_rendering = jnp.broadcast_to(xs_rendering, shape)
            ys_rendering = jnp.broadcast_to(ys_rendering, shape)
            coords_rendering = jnp.stack([xs_rendering, ys_rendering], axis=-1)  # shape (thickness, num_points, 2)
            coords_rendering = jnp.reshape(coords_rendering, (-1, 2)) # shape: (num_points * thickness, 2)
            return coords, coords_rendering

        coords, coords_rendering = jax.lax.cond(condition, calculate, lambda: (jnp.full((num_points, 2), -1, dtype=jnp.int32), jnp.full((num_points * PATH_THICKNESS_HORIZONTAL, 2), -1, dtype=jnp.int32)))

        return coords, coords_rendering 

    def interpolate_vertical_line(edge, condition, num_points=HEIGHT): # Could use less points if performance is a problem, but this accounts for all valid maze changes
        
        def calculate():
            start, end = edge

            # for the computational path mask
            xs = jnp.full((num_points,), start[0], dtype=jnp.int32)  # x-coordinate is constant for vertical lines
            ys = jnp.linspace(start[1], end[1], num_points).astype(jnp.int32)
            coords = jnp.stack([xs, ys], axis=-1)  # shape (num_points, 2)

            # for the rendering path mask
            x_offsets = jnp.arange(PATH_THICKNESS_VERTICAL)  # offsets for the horizontal expansion
            xs_rendering = start[0] + x_offsets[:, None]  # offsets for the horizontal expansion
            ys_rendering = jnp.linspace(start[1]+PATH_THICKNESS_HORIZONTAL, end[1]-1, num_points).astype(jnp.int32) # only render the edges at the parts between the corners, the corners are rendered seperately

            # make sure that the xs and ys are the same shape
            shape = jnp.broadcast_shapes(ys_rendering.shape, xs_rendering.shape)
            xs_rendering = jnp.broadcast_to(xs_rendering, shape)
            ys_rendering = jnp.broadcast_to(ys_rendering, shape)
            coords_rendering = jnp.stack([xs_rendering, ys_rendering], axis=-1)  # shape (thickness, num_points, 2)
            coords_rendering = jnp.reshape(coords_rendering, (-1, 2)) # shape: (num_points * thickness, 2)
            return coords, coords_rendering

        coords, coords_rendering = jax.lax.cond(condition, calculate, lambda: (jnp.full((num_points, 2), -1, dtype=jnp.int32), jnp.full((num_points * PATH_THICKNESS_VERTICAL, 2), -1, dtype=jnp.int32)))

        return coords, coords_rendering 
    
    def render_corner(corner, condition):
        def calculate():
            x, y = corner
            xs = x + jnp.arange(PATH_THICKNESS_VERTICAL)
            ys = y + jnp.arange(PATH_THICKNESS_HORIZONTAL)[:, None]
            shape = jnp.broadcast_shapes(xs.shape, ys.shape)
            xs = jnp.broadcast_to(xs, shape)
            ys = jnp.broadcast_to(ys, shape)
            coords = jnp.stack([xs, ys], axis=-1)
            coords = jnp.reshape(coords, (-1, 2))
            return coords

        coords = jax.lax.cond(condition, calculate, lambda: jnp.full((PATH_THICKNESS_HORIZONTAL*PATH_THICKNESS_VERTICAL, 2), -1, dtype=jnp.int32))
        return coords

    coords_horizontal, rendering_coords_horizontal = jax.vmap(interpolate_horizontal_line, in_axes=(0, 0))(horizontal_edges, horizontal_cond)
    coords_vertical, rendering_coords_vertical = jax.vmap(interpolate_vertical_line, in_axes=(0, 0))(vertical_edges, vertical_cond)
    rendering_coords_corners = jax.vmap(render_corner, in_axes=(0, 0))(corners, corner_cond)

    # flatten the coordinates, because right now they are in the shape (num_lines, num_points, 2) or (num_lines, num_points * PATH_WIDTH_x, 2)
    coords_horizontal = coords_horizontal.reshape(-1, 2)
    rendering_coords_horizontal = rendering_coords_horizontal.reshape(-1, 2)
    coords_vertical = coords_vertical.reshape(-1, 2)
    rendering_coords_vertical = rendering_coords_vertical.reshape(-1, 2)
    rendering_coords_corners = rendering_coords_corners.reshape(-1, 2)

    coords, rendering_coords = jnp.concatenate([coords_horizontal, coords_vertical], axis=0), jnp.concatenate([rendering_coords_horizontal, rendering_coords_vertical, rendering_coords_corners], axis=0)

    # create a mask of valid coordinates in the new shape
    valid_coords = jnp.all(coords >= 0, axis=-1)
    valid_rendering_coords = jnp.all(rendering_coords >= 0, axis=-1)

    # clip the coordinates to the width and height of the mask
    coords = jnp.clip(coords, 0, jnp.array([WIDTH - 1, HEIGHT - 1]))
    rendering_coords = jnp.clip(rendering_coords, 0, jnp.array([WIDTH - 1, HEIGHT - 1]))

    # set the mask values for the horizontal edges
    mask = mask.at[coords[:, 0], coords[:, 1]].add(valid_coords)
    rendering_mask = rendering_mask.at[rendering_coords[:, 0], rendering_coords[:, 1]].add(valid_rendering_coords)

    mask = jnp.clip(mask, 0, 1)  # Ensure the mask values are either 0 or 1
    rendering_mask = jnp.clip(rendering_mask, 0, 1)  # Ensure the rendering mask values are either 0 or 1

    # transpose to match the HWC format for rendering
    rendering_mask = jnp.transpose(rendering_mask, (1, 0))

    return mask, rendering_mask

@partial(jax.jit, static_argnames=['WIDTH', 'HEIGHT'])
def generate_path_pattern(WIDTH, HEIGHT, PATH_COLOR_BROWN, PATH_COLOR_GREEN, WALKED_ON_COLOR):
    """Generates a path pattern for rendering.
    Returns a JAX array of shape (WIDTH, HEIGHT) with the path pattern."""
    # Create an empty mask
    path_pattern_brown = jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8)
    path_pattern_green = jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8)
    walked_on_pattern = jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8)

    # put the indices in a seperate array to be able to use vmap
    ii, jj = jnp.meshgrid(jnp.arange(HEIGHT), jnp.arange(WIDTH), indexing='ij')
    indices = jnp.stack((ii, jj), axis=-1)  # shape (HEIGHT, WIDTH, 2)

    def set_for_column(path_column_brown, path_column_green, walked_on_column, indices):

        def set_color(path_value_brown, path_value_green, walked_on_value, index):
            x, y = index

            path_value_brown, path_value_green, walked_on_value, index = jax.lax.cond(jnp.logical_or(jnp.logical_or(x % 5 == 0, x % 5 == 2), x % 5 == 3),
                lambda: (PATH_COLOR_BROWN, PATH_COLOR_GREEN, walked_on_value, index),
                lambda: (path_value_brown, path_value_green, WALKED_ON_COLOR, index)
            )
            return path_value_brown, path_value_green, walked_on_value, index

        path_column_brown, path_column_green, walked_on_column, index = jax.vmap(set_color, in_axes=0)(path_column_brown, path_column_green, walked_on_column, indices)
        return path_column_brown, path_column_green, walked_on_column, index

    path_pattern_brown, path_pattern_green, walked_on_pattern, _ = jax.vmap(set_for_column, in_axes=0)(path_pattern_brown, path_pattern_green, walked_on_pattern, indices)

    return path_pattern_brown, path_pattern_green, walked_on_pattern

def calculate_corner_rectangles(RECTANGLE_BOUNDS):
    """Calculates which rectangles are at the corners of the maze"""

    # Bounds of the entire maze
    min_x = jnp.min(RECTANGLE_BOUNDS[:, 0])
    min_y = jnp.min(RECTANGLE_BOUNDS[:, 1])
    max_x = jnp.max(RECTANGLE_BOUNDS[:, 2])
    max_y = jnp.max(RECTANGLE_BOUNDS[:, 3])

    def calc(rectangle_bounds):
        """Checks if a rectangle is at two edges of the maze."""
        rect_min_x, rect_min_y, rect_max_x, rect_max_y = rectangle_bounds

        # Check if the rectangle is at the left or right edge
        at_lr_edge = jnp.logical_or(rect_min_x == min_x, rect_max_x == max_x)
        # Check if the rectangle is at the top or bottom edge
        at_tb_edge = jnp.logical_or(rect_min_y == min_y, rect_max_y == max_y)

        return jnp.logical_and(at_lr_edge, at_tb_edge)

    at_two_edges_of_maze = jax.vmap(calc)(RECTANGLE_BOUNDS)

    corners = jnp.nonzero(at_two_edges_of_maze, size=4) # get the indices of the first 4 rectangles which are at at least two edges of the maze --> corners

    # jax.debug.print("Corners found: {corners}", corners=corners)
    return corners

class AmidarConstants(NamedTuple):
    """Constants for the Amidar game. Some constants are precomputed from others to avoid recomputation."""
    # General
    WIDTH: int = 160
    HEIGHT: int = 210
    DIFFICULTY_SETTING: int = 0 # Valid settings are 0 (starts at level 3) and 3 (starts at level 1). If invalid, the game starts at Level 3
    INITIAL_LIVES: int = 3
    MAX_LIVES: int = 3
    FREEZE_DURATION: int = 256  # Duration for which the game is frozen in the beginning and after being hit by an enemy
    CHICKEN_MODE_DURATION: int = 640

    # Deterministic mode 
    DETERMINISTIC_MODE: bool = True
    DETERMINISTIC_KEY: int = jax.random.key(3)

    # Directions
    UP: int = 0 
    LEFT: int = 1
    DOWN: int = 2
    RIGHT: int = 3

    # Rendering
    PATH_COLOR_BROWN = jnp.array([162, 98, 33, 255], dtype=jnp.uint8)  # Brown color for the path
    PATH_COLOR_GREEN = jnp.array([82, 126, 45, 255], dtype=jnp.uint8)  # Green color for the path
    WALKED_ON_COLOR = jnp.array([104, 72, 198, 255], dtype=jnp.uint8)  # Purple color for the walked on paths
    PATH_THICKNESS_HORIZONTAL = chosen_maze.PATH_THICKNESS_HORIZONTAL
    PATH_THICKNESS_VERTICAL = chosen_maze.PATH_THICKNESS_VERTICAL 

    # Points
    PIXELS_PER_POINT_HORIZONTAL: int = 3 # Values to calculate how many points an Edge is worth based on how long it is
    PIXELS_PER_POINT_VERTICAL: int = 30 # Each vertical edge is worth 1 point, since they are 30 pixels long
    BONUS_POINTS_PER_RECTANGLE: int = 48 # Bonus points for completing a rectangle
    BONUS_POINTS_PER_CHICKEN: int = 99 # Bonus points for catching a chicken

    # Player
    PLAYER_SIZE: tuple[int, int] = (7, 7)  # Object sizes (width, height)
    PLAYER_SPRITE_OFFSET: tuple[int, int] = (-1, 0) # Offset for the player sprite in relation to the position in the code (because the top left corner of the player sprite is of the path to the left)
    INITIAL_PLAYER_POSITION: chex.Array = chosen_maze.INITIAL_PLAYER_POSITION
    INITIAL_PLAYER_DIRECTION: chex.Array = UP
    PLAYER_STARTING_PATH = chosen_maze.PLAYER_STARTING_PATH

    # Jumping
    # The jumping mechanics are like this to resemble the ALE version. 
    # There, until frame 477 the jump lasts 30 frames, then it increases and after frame 508 the jumps last 70 frames. 
    # The jump frequency is used to mirror that pressing jump only works every x frames. This increases once at frame 508.
    MAX_JUMPS: int = 4  # Maximum number of jumps the player can perform per life
    INITIAL_JUMP_FREQUENCY: int = 2  # Initial jump frequency (frames)
    JUMP_FREQUENCY: int = 5  # Jump frequency (frames)
    INITIAL_JUMP_DURATION: int = 30  # Initial jump duration (frames)
    JUMP_DURATION: int = 70  # Jump duration after the increase (frames)
    START_JUMP_DURATION_INCREASE: int = 477  # Start of jump duration increase (frames)
    END_JUMP_DURATION_INCREASE: int = 508  # End of jump duration increase (frames)

    # Enemies
    MAX_ENEMIES: int = chosen_maze.MAX_ENEMIES  # Maximum number of enemies on screen
    START_ENEMIES: int = 5  # Number of enemies the lower levels have
    INCREASE_ENEMY_NUMBER_LEVEL: int = 3 # Level at which to switch from START_ENEMIES to MAX_ENEMIES
    ENEMY_SIZE: tuple[int, int] = (7, 7)  # Object sizes (width, height)
    CHICKEN_SIZE: tuple[int, int] = (5, 7)  # Object sizes (width, height)
    ENEMY_SPRITE_OFFSET: tuple[int, int] = (-1, 0) # Offset for the enemy sprite in relation to the position in the code (because the top left corner of the enemy sprite is of the path to the left)
    INITIAL_ENEMY_POSITIONS: chex.Array = chosen_maze.INITIAL_ENEMY_POSITIONS
    INITIAL_ENEMY_DIRECTIONS: chex.Array = jnp.array([RIGHT] * MAX_ENEMIES)  # All enemies start moving right
    # Enemy Types
    SHADOW: int = 0 
    WARRIOR: int = 1  
    PIG: int = 2
    CHICKEN: int = 3
    INVALID_ENEMY: int = -1

    # Path Structure
    PATH_CORNERS: chex.Array = chosen_maze.PATH_CORNERS
    HORIZONTAL_PATH_EDGES: chex.Array = chosen_maze.HORIZONTAL_PATH_EDGES
    VERTICAL_PATH_EDGES: chex.Array = chosen_maze.VERTICAL_PATH_EDGES
    PATH_EDGES: chex.Array = chosen_maze.PATH_EDGES
    RECTANGLES: chex.Array = chosen_maze.RECTANGLES
    RECTANGLE_BOUNDS: chex.Array = chosen_maze.RECTANGLE_BOUNDS
    CORNER_RECTANGLES: chex.Array = chosen_maze.CORNER_RECTANGLES
    SHORT_PATHS: chex.Array = chosen_maze.SHORT_PATHS # Array of [corner_index, corner_index, edge_index] Short paths are paths between corners that are directly next to each other. These are marked as walked on once both it's corners have been walked on, even if one hasn't walked between them. 

    # Precomputed Constants
    # Path/Rendering
    PATH_MASK, RENDERING_PATH_MASK = generate_path_mask(WIDTH, HEIGHT, PATH_THICKNESS_HORIZONTAL, PATH_THICKNESS_VERTICAL, HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES, PATH_CORNERS, jnp.full((HORIZONTAL_PATH_EDGES.shape[0],), True), jnp.full((VERTICAL_PATH_EDGES.shape[0],), True), jnp.full((PATH_CORNERS.shape[0],), True))  # Path mask are the single lines which restrict the movement, while rendering path mask includes the width of the paths for rendering
    PATH_PATTERN_BROWN, PATH_PATTERN_GREEN, WALKED_ON_PATTERN = generate_path_pattern(WIDTH, HEIGHT, PATH_COLOR_BROWN, PATH_COLOR_GREEN, WALKED_ON_COLOR)
    PATH_SPRITE_BROWN: chex.Array = jnp.where(RENDERING_PATH_MASK[:, :, None] == 1, PATH_PATTERN_BROWN, jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8))
    PATH_SPRITE_GREEN: chex.Array = jnp.where(RENDERING_PATH_MASK[:, :, None] == 1, PATH_PATTERN_GREEN, jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8))
    
# immutable state container
class AmidarState(NamedTuple):
    frame_counter: chex.Array
    random_key: chex.Array  # Random key for JAX operations
    freeze_counter: chex.Array  # Counter for freezing the game
    level: chex.Array
    score: chex.Array
    lives: chex.Array
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    last_walked_corner: chex.Array # (2,) -> (x, y) of the last corner walked on
    walked_on_paths: chex.Array
    walked_on_corners: chex.Array
    completed_rectangles: chex.Array
    enemy_positions: chex.Array # (MAX_ENEMIES, 2) -> (x, y) for each enemy
    enemy_directions: chex.Array # (MAX_ENEMIES, 1) -> direction of each enemy
    enemy_types: chex.Array # (MAX_ENEMIES, 1) -> type of each enemy (Shadow, Warrior, Pig, Chicken)
    chicken_counter: chex.Array  # Counter for the chicken mode, counts down from CHICKEN_MODE_DURATION
    jump_counter: chex.Array
    times_jumped: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class AmidarObservation(NamedTuple):
    player_gorilla: EntityPosition
    player_paint_roller: EntityPosition
    shadows: chex.Array
    warriors: chex.Array
    pigs: chex.Array
    chickens: chex.Array
    lives: chex.Array
    paths: chex.Array
    walked_on_paths: chex.Array
    completed_rectangles: chex.Array

class AmidarInfo(NamedTuple):
    level: jnp.ndarray
    frame_counter: jnp.ndarray
    all_rewards: chex.Array

def get_player_speed(frame_counter: chex.Array) -> chex.Array:
    """
    Compute the player speed based on the frame counter.
    Corresponds to whether the player is moving this frame or not.

    The player speed is based on a repeating '01001' pattern with periodic skips.

    The base sequence repeats every 5 positions as:
        0, 1, 0, 0, 1
    which means '1' occurs only at indices where n % 5 == 1 or n % 5 == 4.

    Certain '1' positions are periodically replaced with '0' according to
    two skip positions r1 and r2 modulo 155 (where 155 = 5 × 31). These
    skip positions alternate every 78 and 77 steps, creating a full cycle
    of length 155.

    r1 and r2 are chosen such that r1 % 5 and r2 % 5 are both in {1, 4}.
    In the ALE game, these are different each game, but what the desicion is based on is not clear, 
    so for now they are just set to 66 and 144.
    Here are all the possible combinations of r1 and r2:
    [[1, 79], [4, 81], [6, 84], [9, 86], [11, 89], [14, 91], [16, 94], [19, 96], [21, 99], [24, 101], 
    [26, 104], [29, 106], [31, 109], [34, 111], [36, 114], [39, 116], [41, 119], [44, 121], [46, 124], 
    [49, 126], [51, 129], [54, 131], [56, 134], [59, 136], [61, 139], [64, 141], [66, 144], [69, 146], 
    [71, 149], [74, 151], [76, 154]]


    Parameters
    ----------
    frame_counter : int 
        The frame in which to check if the player moves.
    
    Returns
    -------
    jax.numpy.ndarray
        Boolean of whether the player moves at the given frame.

    Notes
    -----
    - The base pattern is defined purely by n % 5.
    - The skip mask is defined by n % 155 ∈ {r1, r2}.
    - Multiplying the base pattern by the skip mask produces the final sequence.
    """
    frame_counter = jnp.asarray(frame_counter)
    r1, r2 = 66, 144

    # Base pattern: 1 if n % 5 == 1 or 4
    base = jnp.isin(frame_counter % 5, jnp.array([1, 4], dtype=frame_counter.dtype))

    # Skip mask: 0 at r1 or r2 modulo 155, else 1
    skip_mask = jnp.isin(frame_counter % 155, jnp.array([r1 % 155, r2 % 155], dtype=frame_counter.dtype), invert=True)

    return jnp.logical_and(base, skip_mask).astype(jnp.int32)

def get_enemy_speed(frame_counter: chex.Array, level: chex.Array) -> chex.Array:
    """
    Gets the enemy speed based on the frame counter and level.
    The speed stays the same after level 7 (tested until level 10)
    """
    compute_for_level_functions = [
        lambda: jnp.isin(frame_counter % 5, jnp.array([0, 3])), # level 1
        lambda: jnp.isin(frame_counter % 25, jnp.array([2, 4, 7, 9, 12, 14, 16, 17, 19, 22, 24])), # level 2
        lambda: jnp.isin(frame_counter % 25, jnp.array([0, 2, 3, 5, 8, 10, 13, 15, 17, 18, 20, 23])), # level 3
        lambda: jnp.isin(frame_counter % 25, jnp.array([1, 4, 6, 8, 9, 11, 14, 16, 18, 19, 21, 24])), # level 4
        lambda: jnp.isin(frame_counter % 25, jnp.array([0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22])), # level 5
        lambda: jnp.isin(frame_counter % 25, jnp.array([0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22])), # level 6
        lambda: jnp.isin(frame_counter % 5, jnp.array([0, 2, 3])), # level 7
    ]

    # level-1 since levels are 1-indexed, but jax.lax.switch is 0-indexed
    enemy_speed = jax.lax.switch(level-1, compute_for_level_functions)

    return enemy_speed


def player_step(constants: AmidarConstants, state: AmidarState, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Updates the player position based on the action taken.
    Returns the new player x and y coordinates and the direction."""

    def on_path(x, y):
        """Checks if the given coordinates are on the path."""
        # add 1 to x and y to account for the offset of the top left corner of the player sprite in relation to the path mask
        return constants.PATH_MASK[x, y] == 1

    up = jnp.logical_or( action == Action.UP, action == Action.UPFIRE)
    down = jnp.logical_or( action == Action.DOWN, action == Action.DOWNFIRE)
    left = jnp.logical_or( action == Action.LEFT, action == Action.LEFTFIRE)
    right = jnp.logical_or( action == Action.RIGHT, action == Action.RIGHTFIRE)

    new_x = state.player_x + jnp.where(left, -1, 0) + jnp.where(right, 1, 0)
    new_y = state.player_y + jnp.where(up, -1, 0) + jnp.where(down, 1, 0)

    new_x = jnp.where(on_path(new_x, new_y), new_x, state.player_x)
    new_y = jnp.where(on_path(new_x, new_y), new_y, state.player_y)

    # if the direction is not possiple to move in, try moving in the previous direction
    def move_in_previous_direction(direction):
        new_x = state.player_x + jnp.where(direction == constants.LEFT, -1, 0) + jnp.where(direction == constants.RIGHT, 1, 0)
        new_y = state.player_y + jnp.where(direction == constants.UP, -1, 0) + jnp.where(direction == constants.DOWN, 1, 0)
        # only move if new position is on the path
        new_x = jnp.where(on_path(new_x, new_y), new_x, state.player_x)
        new_y = jnp.where(on_path(new_x, new_y), new_y, state.player_y)
        return new_x, new_y
    
    has_not_moved = jnp.logical_and(new_x == state.player_x, new_y == state.player_y)
    movement_key_pressed = jnp.logical_or(up, jnp.logical_or(down, jnp.logical_or(left, right)))
    new_x, new_y = jax.lax.cond(jnp.logical_and(has_not_moved, movement_key_pressed), move_in_previous_direction, lambda direction: (new_x, new_y), state.player_direction)
        
    player_direction = jnp.select([new_y < state.player_y, new_x < state.player_x, new_y > state.player_y, new_x > state.player_x],
                                  [constants.UP,           constants.LEFT,         constants.DOWN,         constants.RIGHT       ], 
                                  default=state.player_direction)

    # Check if the new position is a corner, in which case check if a new path edge is walked on

    def corner_handeling():
        # Set the walked on paths
        # last walked corner -> (new_x, new_y)

        # set the corner as walked on
        corner_mask = jnp.all(constants.PATH_CORNERS == jnp.array([new_x, new_y]), axis=1)
        corner_index = jnp.argmax(corner_mask).astype(jnp.int32)
        walked_on_corners = state.walked_on_corners.at[corner_index].set(1)

        def find_edge(edge, edges):
            # edge: shape (2, 2)
            # edges: shape (N, 2, 2)
            match_forward = jnp.all(edges == edge, axis=(1, 2))
            match_reverse = jnp.all(edges == edge[::-1], axis=(1, 2))
            matches = jnp.logical_or(match_forward, match_reverse)
            # Get the index of the first match, or -1 if not found
            index = jnp.argmax(matches)
            found = jnp.any(matches)
            index = jnp.where(found, index, -1)
            return index

        def score_points(edge_index, walked_on_paths, completed_rectangles):
            """Scores points based on the edge walked on."""
            # Calculate the points scored based on the edge walked on
            edge = constants.PATH_EDGES[edge_index]
            points_scored = jax.lax.cond(edge[0, 0] == edge[1, 0],  # Vertical edge
                         lambda: ((edge[1, 1] - edge[0, 1]) // constants.PIXELS_PER_POINT_VERTICAL),  # Vertical edge
                         lambda: ((edge[1, 0] - edge[0, 0]) // constants.PIXELS_PER_POINT_HORIZONTAL))  # Horizontal edge
            # Ensure points_scored is an integer
            points_scored = points_scored.astype(jnp.int32)
            # Update the walked on paths
            walked_on_paths = walked_on_paths.at[edge_index].set(1)

            def check_rectangle_completion(completed_rectangles):
                """Checks if a rectangle is completed and updates the completed rectangles."""
                # Create a mask for rectangles that contain this edge
                rectangles_containing_edge_mask = constants.RECTANGLES[:, edge_index] == 1

                # Check if any rectangles containing this edge are now completed
                def check_single_rectangle(rect_idx):
                    # Get the rectangle (which edges it contains)
                    rectangle = constants.RECTANGLES[rect_idx]
                    # Check if all edges in this rectangle have been walked on
                    all_edges_walked = jnp.all(jnp.where(rectangle == 1, walked_on_paths, True))
                    return all_edges_walked
                
                # Check all rectangles to see if they're completed
                all_rectangle_indices = jnp.arange(constants.RECTANGLES.shape[0])
                completed_mask = jax.vmap(check_single_rectangle)(all_rectangle_indices)
                
                # Only consider rectangles that contain this edge
                relevant_completed_mask = jnp.logical_and(rectangles_containing_edge_mask, completed_mask)
                
                # Check if any new rectangles are completed (not already marked as completed)
                new_completions = jnp.logical_and(relevant_completed_mask, 
                                                 jnp.logical_not(completed_rectangles))
                
                new_rectangle_completed = jnp.any(new_completions)
                
                # Update completed rectangles
                completed_rectangles = jnp.logical_or(completed_rectangles, new_completions)
                
                return new_rectangle_completed, completed_rectangles
            
            # call check_rectangle_completion to check if a rectangle is completed
            new_rectangle_completed, completed_rectangles = check_rectangle_completion(completed_rectangles)
            # Add bonus points for completing a rectangle
            points_scored = jax.lax.cond(new_rectangle_completed, lambda: points_scored + constants.BONUS_POINTS_PER_RECTANGLE, lambda: points_scored)

            return points_scored, walked_on_paths, completed_rectangles

        # find the edge that has just been walked
        match_index = find_edge(jnp.array([[new_x, new_y], state.last_walked_corner]), constants.PATH_EDGES)
        # If the edge is not walked on yet, score points and update the walked on paths & completed rectangles
        points_scored, walked_on_paths, completed_rectangles = jax.lax.cond(jnp.logical_and(match_index >= 0, state.walked_on_paths[match_index] == 0), lambda: score_points(match_index, state.walked_on_paths, state.completed_rectangles), lambda: (0, state.walked_on_paths, state.completed_rectangles))

        # Process short paths sequentially and thread updates
        def process_short_path(carry, short_path):
            points_acc, walked_on_paths, completed_rectangles = carry
            cond = jnp.logical_and(jnp.logical_and(walked_on_corners[short_path[0]] == 1, walked_on_corners[short_path[1]] == 1), walked_on_paths[short_path[2]] == 0)

            def take():
                p, wop, crect = score_points(short_path[2], walked_on_paths, completed_rectangles)
                return (points_acc + p, wop, crect), None

            def skip():
                return (points_acc, walked_on_paths, completed_rectangles), None

            return jax.lax.cond(cond, take, skip)

        (points_scored, walked_on_paths, completed_rectangles), _ = jax.lax.scan(process_short_path, (points_scored, walked_on_paths, completed_rectangles), constants.SHORT_PATHS)

        corners_completed = jnp.all(completed_rectangles[constants.CORNER_RECTANGLES])  # Check if all corners are completed

        next_level = jnp.all(walked_on_paths)  # Check if all paths are walked on

        last_walked_corner = jnp.array([new_x, new_y])   # Update the last walked corner to the new position
        return points_scored, last_walked_corner, walked_on_paths, walked_on_corners, completed_rectangles, corners_completed, next_level


    is_corner = jnp.any(jnp.all(constants.PATH_CORNERS == jnp.array([new_x, new_y]), axis=1))
    points_scored, last_walked_corner, walked_on_paths, walked_on_corners, completed_rectangles, corners_completed, next_level = jax.lax.cond(is_corner, corner_handeling, lambda: (0, state.last_walked_corner, state.walked_on_paths, state.walked_on_corners, state.completed_rectangles, False, False))

    return points_scored, new_x, new_y, player_direction, last_walked_corner, walked_on_paths, walked_on_corners, completed_rectangles, corners_completed, next_level

def enemies_step(constants: AmidarConstants, state: AmidarState, random_key: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Updates the enemy positions based on their behavior."""

    # since this shape is constant, using normal if is okay. During compilation, the correct path is chosen and the other one is not compiled at all.
    if state.enemy_positions.shape[0] == 0:
        return state.enemy_positions, state.enemy_directions

    enemy_keys = jax.random.split(random_key, state.enemy_positions.shape[0])  # Split the random key for each enemy

    # Calculate possible movement directions for enemies
    def calculate_possible_enemy_directions(enemy_position):
        """Calculates possible movement directions for an enemy."""
        # 0=up, 1=left, 2=down, 3=right
        x, y = enemy_position
        up = constants.PATH_MASK[x, y - 1]
        down = constants.PATH_MASK[x, y + 1]
        left = constants.PATH_MASK[x - 1, y]
        right = constants.PATH_MASK[x + 1, y]
        return jnp.array([up, left, down, right])

    possible_directions_per_enemy = jax.vmap(calculate_possible_enemy_directions, in_axes=0)(state.enemy_positions)

    def calculate_good_enemy_directions(enemy_direction, possible_directions, index):
        """From the possible movement directions exclude u-turns and the direction the enemy is currently moving in.
        The first enemy(index 0) is the Tracer and only moves along the perimeter, so it's good direction is the current one."""
        opposite_direction = (enemy_direction + 2) % 4
        possible_directions_without_u_turns = possible_directions.at[opposite_direction].set(False) # Exclude the opposite direction (u-turn) 
        good_directions = jax.lax.cond(index == 0,
                                        lambda: possible_directions_without_u_turns & jnp.full(possible_directions.shape, False).at[enemy_direction].set(True),  # Tracer should always go in the current direction
                                        lambda: possible_directions_without_u_turns.at[enemy_direction].set(False))  # every other enemy should take every turn it can
        return good_directions, possible_directions_without_u_turns

    good_directions_per_enemy, possible_directions_without_u_turns_per_enemy = jax.vmap(calculate_good_enemy_directions, in_axes=0)(state.enemy_directions, possible_directions_per_enemy, jnp.arange(state.enemy_directions.shape[0]))

    # jax.debug.print("good: \n {}, \n okay: \n {}, \n possible: \n {}", good_directions_per_enemy, possible_directions_without_u_turns_per_enemy, possible_directions_per_enemy)

    def move_enemy(enemy_x, enemy_y, direction, possible_directions, possible_directions_without_u_turns, good_directions, random_key, index):
        """Makes the enemy move according to it's movement pattern."""

        def choose_direction():
            # If there are good directions, choose one, else choose from all non-u-turns, if that is not possible, choose from all possible directions
            chosen_direction = jax.lax.cond(jnp.any(good_directions),
                                            lambda: jax.random.choice(random_key, jnp.arange(4), shape=(), p=jnp.where(good_directions, 1/jnp.count_nonzero(good_directions), 0)),
                                            lambda: jax.lax.cond(jnp.any(possible_directions_without_u_turns),
                                                             lambda: jax.random.choice(random_key, jnp.arange(4), shape=(), p=jnp.where(possible_directions_without_u_turns, 1/jnp.count_nonzero(possible_directions_without_u_turns), 0)),
                                                             lambda: jax.random.choice(random_key, jnp.arange(4), shape=(), p=jnp.where(possible_directions, 1/jnp.count_nonzero(possible_directions), 0))
                                                            )
                                            )  
            # the tracer has to turn left if it can, otherwise it will deviate from the border 
            left_turn_direction = (direction + 1) % 4
            chosen_direction = jax.lax.cond(jnp.logical_and(index == 0, possible_directions_without_u_turns[left_turn_direction]), lambda: left_turn_direction, lambda: chosen_direction)
            return chosen_direction

        # if no directions are possible stay with the current one
        chosen_direction = jax.lax.cond(jnp.any(possible_directions), choose_direction, lambda: direction)
        
        new_x, new_y = jax.lax.cond(jnp.any(possible_directions),
            lambda: (enemy_x + jnp.where(chosen_direction == constants.LEFT, -1, 0) + jnp.where(chosen_direction == constants.RIGHT, 1, 0),
                     enemy_y + jnp.where(chosen_direction == constants.UP, -1, 0) + jnp.where(chosen_direction == constants.DOWN, 1, 0)),
            lambda: (enemy_x, enemy_y))  # If the direction is not possible, stay in place
        return jnp.array([new_x, new_y]), chosen_direction

    new_enemy_positions, new_enemy_directions = jax.vmap(move_enemy, in_axes=0)(state.enemy_positions[:, 0], state.enemy_positions[:, 1], state.enemy_directions, possible_directions_per_enemy, possible_directions_without_u_turns_per_enemy, good_directions_per_enemy, enemy_keys, jnp.arange(state.enemy_directions.shape[0]))

    return new_enemy_positions, new_enemy_directions

def chicken_mode(constants, level, corners_completed, enemy_types, chicken_counter, jump_counter):
    # If the chicken counter is 0 deactivate the chickens
    enemy_types = jax.lax.cond(chicken_counter == 0, lambda: get_enemy_types(constants, level), lambda: enemy_types)
    # If the chicken counter is greater or equal to 0 and less than the maximum, decrement it
    chicken_counter, chicken_mode = jax.lax.cond(jnp.logical_and(chicken_counter >= 0, chicken_counter < constants.CHICKEN_MODE_DURATION), lambda: (chicken_counter - 1, True), lambda: (chicken_counter, False))
    # If all corners are completed and the chicken counter is still at the maximum, activate the chickens (any jump ends immediately)
    enemy_types, chicken_counter, jump_counter = jax.lax.cond(jnp.logical_and(corners_completed, chicken_counter == constants.CHICKEN_MODE_DURATION), lambda: (jnp.where(enemy_types != constants.INVALID_ENEMY, constants.CHICKEN, enemy_types), chicken_counter-1, -1), lambda: (enemy_types, chicken_counter, jump_counter))
    # jax.debug.print("Enemy types: {enemy_types}", enemy_types=enemy_types)
    # jax.lax.cond(corners_completed, lambda: jax.debug.print("Corners completed, Chicken Counter{}", chicken_counter), lambda: None)
    return enemy_types, chicken_counter, chicken_mode, jump_counter

def jump(constants, level, frame_counter, action, jump_counter, chicken_mode_active, times_jumped, enemy_types):
    """
    jump counter stays at -1
    if the player presses jump and the times jumped < 4 and not chicken mode and not already jumping
    jump counter is set to current jump duration, enemies -> shadows
    every frame the jump counter is reduced until it reaches -1, at 0 shadows -> enemies
    """

    def get_jump_duration():
        # Before START_JUMP_DURATION_INCREASE: initial duration
        # Between START_JUMP_DURATION_INCREASE and END_JUMP_DURATION_INCREASE: linearly increase
        # After END_JUMP_DURATION_INCREASE: final duration

        def linear_increase():
            # Linearly interpolate between INITIAL_JUMP_DURATION and JUMP_DURATION
            progress = (frame_counter - constants.START_JUMP_DURATION_INCREASE) / (constants.END_JUMP_DURATION_INCREASE - constants.START_JUMP_DURATION_INCREASE)
            duration = constants.INITIAL_JUMP_DURATION + progress * (constants.JUMP_DURATION - constants.INITIAL_JUMP_DURATION)
            return duration.astype(jnp.int32)

        duration = jax.lax.cond(
            frame_counter < constants.START_JUMP_DURATION_INCREASE,
            lambda: constants.INITIAL_JUMP_DURATION,
            lambda: jax.lax.cond(
                frame_counter < constants.END_JUMP_DURATION_INCREASE,
                linear_increase,
                lambda: constants.JUMP_DURATION
            )
        )
        return duration

    jump_action = jnp.any(jnp.array([action == Action.FIRE, action == Action.UPFIRE, action == Action.LEFTFIRE, action == Action.DOWNFIRE, action == Action.RIGHTFIRE]))
    # the player can only jump every other frame in the beginning, then when the full jump length is reached every 5 frames
    timing_condition = jnp.logical_or(jnp.logical_and(frame_counter <= constants.END_JUMP_DURATION_INCREASE, frame_counter % constants.INITIAL_JUMP_FREQUENCY == 0), jnp.logical_and(frame_counter > constants.END_JUMP_DURATION_INCREASE, frame_counter % constants.JUMP_FREQUENCY == 0))
    can_jump = jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(times_jumped < constants.MAX_JUMPS, timing_condition), jnp.bitwise_not(chicken_mode_active)), jump_counter < 0), jump_action)  # Check if the player can jump
    # If the player can jump, set the jump counter to the jump duration and increment the times jumped, turn enemies into shadows
    enemy_types, jump_counter, times_jumped = jax.lax.cond(can_jump, lambda: (jnp.where(enemy_types != constants.INVALID_ENEMY, constants.SHADOW, enemy_types), get_jump_duration(), times_jumped + 1), lambda: (enemy_types, jump_counter, times_jumped))
    enemy_types = jax.lax.cond(jump_counter == 0, lambda: get_enemy_types(constants, level), lambda: enemy_types)
    jump_counter = jax.lax.cond(jump_counter >= 0, lambda: jump_counter - 1, lambda: jump_counter)  # Decrement the jump counter if it is greater than 0
    # jax.lax.cond(jnp.any(enemy_types == constants.SHADOW), lambda: jax.debug.print("Jump activated, {}", jump_counter), lambda: None)
    return enemy_types, jump_counter, times_jumped


def check_for_collisions(constants, player_x, player_y, enemy_positions, enemy_types) -> bool:
    """Checks if the player collides with any enemy."""
    # the collision is counted if any part of the sprites touch, so per enemy we need to check if the sprite overlaps the players

    # since this shape is constant, using normal if is okay. During compilation, the correct path is chosen and the other one is not compiled at all.
    if enemy_positions.shape[0] == 0:
        return jnp.zeros((0,), dtype=jnp.bool_)

    def check_enemy_for_collision(enemy_x, enemy_y, enemy_type):
        """Checks if the enemy collides with the player."""
        collision = jax.lax.cond(
            jnp.logical_or(enemy_type == 0, enemy_type == -1), # shadows and invalid enemies don't collide with the player
            lambda: False,
            lambda: jnp.logical_not(
                        (enemy_x + constants.ENEMY_SIZE[0] <= player_x)    |   # enemy is to the left of the player
                        (player_x + constants.PLAYER_SIZE[0] <= enemy_x)   |   # enemy is to the right of the player
                        (enemy_y + constants.ENEMY_SIZE[1] <= player_y)    |   # enemy is above the player
                        (player_y + constants.PLAYER_SIZE[1] <= enemy_y)       # enemy is below the player
                    )
        )
        return collision

    return jax.vmap(check_enemy_for_collision, in_axes=0)(enemy_positions[:, 0], enemy_positions[:, 1], enemy_types)

def handle_collisions(constants, enemy_types: chex.Array, collisions: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Handles the collisions with the enemies.
    Returns the new enemy positions, directions, types, lives and points scored."""

    def handle_collision(enemy_type, collision):
        """Handles the collision with a single enemy."""
        
        def collide(): 
            return jax.lax.cond(enemy_type == 3,  # Chicken
                                lambda: (constants.SHADOW, constants.BONUS_POINTS_PER_CHICKEN, 0),  # Chicken gives bonus points and does not cost a life
                                lambda: (enemy_type, 0, 1))

        enemy_type, points, lost_live = jax.lax.cond(collision, collide, lambda: (enemy_type, 0, 0))
        return enemy_type, points, lost_live

    enemy_types, points_scored, lost_live = jax.vmap(handle_collision, in_axes=(0, 0))(enemy_types, collisions)

    lost_live = jnp.any(lost_live)  # If any enemy collision results in a lost life, set lost_live to True
    points_scored = jnp.sum(points_scored)  # Sum the points scored from all collisions

    return enemy_types, points_scored, lost_live

def get_enemy_types(constants, level: chex.Array) -> chex.Array: 
    enemy_type_this_level = jax.lax.cond(level % 2 == 0, lambda: constants.PIG, lambda: constants.WARRIOR)
    enemies = jnp.full_like(constants.INITIAL_ENEMY_DIRECTIONS, enemy_type_this_level, dtype=jnp.int32)
    mask = jnp.arange(enemies.shape[0]) >= constants.START_ENEMIES
    enemies = jax.lax.cond(level < constants.INCREASE_ENEMY_NUMBER_LEVEL, lambda: jnp.where(mask, constants.INVALID_ENEMY, enemies), lambda: enemies,)
    return enemies

def activate_next_level(constants, reset_key, level, lives, enemy_positions, enemy_directions, enemy_types):
    level = level + 1
    lives = jnp.minimum(lives + 1, constants.MAX_LIVES)
    enemy_positions = constants.INITIAL_ENEMY_POSITIONS
    enemy_directions = constants.INITIAL_ENEMY_DIRECTIONS
    enemy_types = get_enemy_types(constants, level)  # Get the enemy types for the new level
    freeze_counter = constants.FREEZE_DURATION  # Reset the freeze counter for the next level
    walked_on_paths = (jnp.zeros(jnp.shape(constants.PATH_EDGES)[0], dtype=jnp.int32)).at[constants.PLAYER_STARTING_PATH].set(1)
    walked_on_corners = jnp.zeros(jnp.shape(constants.PATH_CORNERS)[0], dtype=jnp.int32)
    completed_rectangles = jnp.zeros(jnp.shape(constants.RECTANGLES)[0], dtype=jnp.bool_)
    player_x = constants.INITIAL_PLAYER_POSITION[0]
    player_y = constants.INITIAL_PLAYER_POSITION[1]
    player_direction = constants.INITIAL_PLAYER_DIRECTION
    chicken_counter = jnp.array(constants.CHICKEN_MODE_DURATION).astype(jnp.int32)  # Reset chicken counter
    jump_counter = -1
    times_jumped = 0

    return reset_key, level, lives, enemy_positions, enemy_directions, enemy_types, freeze_counter, walked_on_paths, walked_on_corners, completed_rectangles, player_x, player_y, player_direction, chicken_counter, jump_counter, times_jumped


class JaxAmidar(JaxEnvironment[AmidarState, AmidarObservation, AmidarInfo, AmidarConstants]):
    def __init__(self, constants: AmidarConstants = None, reward_funcs: list[callable]=None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPFIRE,
            Action.DOWNFIRE,
        ]
        self.constants = constants or AmidarConstants()
        self.obs_size = 4 #TODO add as needed
        self.renderer = AmidarRenderer(constants)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AmidarState) -> bool:
        return state.lives <= 0
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: AmidarState, state: AmidarState):
        return state.score - previous_state.score
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: AmidarState, state: AmidarState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AmidarState, all_rewards: jnp.ndarray) -> AmidarInfo:
        return AmidarInfo(
            level=state.level,
            frame_counter=state.frame_counter,
            all_rewards=all_rewards,
        )

    def reset(self, key = jax.random.key(3)) -> Tuple[AmidarObservation, AmidarState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        key = jax.random.wrap_key_data(key)
        level = jax.lax.cond(self.constants.DIFFICULTY_SETTING == 3, lambda: jnp.array(1).astype(jnp.int32), lambda: jnp.array(3).astype(jnp.int32))
        state = AmidarState(
            frame_counter=jnp.array(0).astype(jnp.int32),  # Frame counter for the game
            random_key=jax.lax.cond(self.constants.DETERMINISTIC_MODE, lambda: self.constants.DETERMINISTIC_KEY, lambda: key),
            freeze_counter=jnp.array(self.constants.FREEZE_DURATION).astype(jnp.int32),  # Freeze counter for the initial freeze
            level=level,
            score=jnp.array(0).astype(jnp.int32),  # Initial score
            lives=jnp.array(self.constants.INITIAL_LIVES).astype(jnp.int32),  # Initial lives
            player_x=jnp.array(self.constants.INITIAL_PLAYER_POSITION[0]).astype(jnp.int32),
            player_y=jnp.array(self.constants.INITIAL_PLAYER_POSITION[1]).astype(jnp.int32),
            player_direction=jnp.array(self.constants.INITIAL_PLAYER_DIRECTION).astype(jnp.int32),
            last_walked_corner=jnp.array([0, 0]).astype(jnp.int32),  # Last corner walked on
            walked_on_paths=(jnp.zeros(jnp.shape(self.constants.PATH_EDGES)[0], dtype=jnp.int32)).at[self.constants.PLAYER_STARTING_PATH].set(1),  # Initialize walked on paths
            walked_on_corners=jnp.zeros(jnp.shape(self.constants.PATH_CORNERS)[0], dtype=jnp.int32),  # Initialize walked on corners
            completed_rectangles=jnp.zeros(jnp.shape(self.constants.RECTANGLES)[0], dtype=jnp.bool_),  # Initialize completed rectangles
            enemy_positions=self.constants.INITIAL_ENEMY_POSITIONS,
            enemy_directions=self.constants.INITIAL_ENEMY_DIRECTIONS,
            enemy_types=get_enemy_types(self.constants, level),
            chicken_counter=jnp.array(self.constants.CHICKEN_MODE_DURATION).astype(jnp.int32),  # Initial chicken counter
            jump_counter=jnp.array(-1).astype(jnp.int32),  # Initial jump counter
            times_jumped=jnp.array(0).astype(jnp.int32), 
        )
        initial_obs = self._get_observation(state)

        # jnp.set_printoptions(threshold=jnp.inf)
        # jax.debug.print("{m}", m=PATH_MASK)

        return initial_obs, state
    

    def observation_space(self) -> spaces.Box:
        """ 
        Returns the observation space for Amidar.
        The observation contains:
        - player_gorilla: PlayerEntity (x, y, width, height, active)
        - player_paint_roller: PlayerEntity (x, y, width, height, active)
        - shadows: array of shape (MAX_ENEMIES, 5) with x,y,width,height,active for each shadow
        - warriors: array of shape (MAX_ENEMIES, 5) with x,y,width,height,active for each warrior
        - pigs: array of shape (MAX_ENEMIES, 5) with x,y,width,height,active for each pig
        - chickens: array of shape (MAX_ENEMIES, 5) with x,y,width,height,active for each chicken
        - lives: int (0-MAX_LIVES)
        - paths: array of shape (num_edges, 5) with x,y,width,height,active for each path
        - walked_on_paths: array of shape (num_edges, 5) with x,y,width,height,active for each path (active when walked on)
        - completed_rectangles: array of shape (num_rectangles, 5) with x,y,width,height,active for each rectangle (active when completed)
        """
        HEIGHT = self.constants.HEIGHT
        WIDTH = self.constants.WIDTH
        MAX_ENEMIES = self.constants.MAX_ENEMIES
        path_shape = self.constants.PATH_EDGES.shape[0]

        def enemy_box(enemy_size_wh):
            # per-entity [x, y, w, h, active] bounds
            hi_vec = jnp.array([WIDTH, HEIGHT, enemy_size_wh[0], enemy_size_wh[1], 1], dtype=jnp.int32)
            lo_vec = jnp.zeros_like(hi_vec)
            high = jnp.broadcast_to(hi_vec, (MAX_ENEMIES, 5))
            low = jnp.broadcast_to(lo_vec, (MAX_ENEMIES, 5))
            return spaces.Box(low=low, high=high, shape=(MAX_ENEMIES, 5), dtype=jnp.int32)

        def wha_box(n):
            # generic [x, y, w, h, active] for paths/rectangles
            hi_vec = jnp.array([WIDTH, HEIGHT, WIDTH, HEIGHT, 1], dtype=jnp.int32)
            lo_vec = jnp.zeros_like(hi_vec)
            high = jnp.broadcast_to(hi_vec, (n, 5))
            low = jnp.broadcast_to(lo_vec, (n, 5))
            return spaces.Box(low=low, high=high, shape=(n, 5), dtype=jnp.int32)


        return spaces.Dict({
            "player_gorilla": spaces.Dict({
                "x": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_paint_roller": spaces.Dict({
                "x": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "shadows": enemy_box(self.constants.ENEMY_SIZE),
            "warriors": enemy_box(self.constants.ENEMY_SIZE),
            "pigs": enemy_box(self.constants.ENEMY_SIZE),
            "chickens": enemy_box(self.constants.CHICKEN_SIZE),
            "lives": spaces.Box(low=0, high=self.constants.MAX_LIVES, shape=(), dtype=jnp.int32),
            "paths": wha_box(path_shape),
            "walked_on_paths": wha_box(path_shape),
            "completed_rectangles": wha_box(self.constants.RECTANGLES.shape[0]),
        })

    def state_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.constants.HEIGHT, self.constants.WIDTH, 4), dtype=jnp.uint8)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    
    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([jnp.array([entity.x]), jnp.array([entity.y]), jnp.array([entity.width]), jnp.array([entity.height]), jnp.array([entity.active])])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AmidarObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_entity_position(obs.player_gorilla),
            self.flatten_entity_position(obs.player_paint_roller),
            obs.shadows.flatten(),
            obs.warriors.flatten(),
            obs.pigs.flatten(),
            obs.chickens.flatten(),
            obs.lives.flatten(),
            obs.paths.flatten(),
            obs.walked_on_paths.flatten(),
            obs.completed_rectangles.flatten(),
        ])

    def image_space(self) -> spaces.Box:
        """
        Returns the image space of the renderer output.
        Shape is (HEIGHT, WIDTH, 3) with uint8 values in [0, 255].
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.constants.HEIGHT, self.constants.WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AmidarState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AmidarState, action: chex.Array) -> Tuple[AmidarObservation, AmidarState, float, bool, AmidarInfo]:

        previous_state = state
            
        def take_step():
            player_state = jax.lax.cond(get_player_speed(state.frame_counter), player_step, lambda constants, state, action: (0, state.player_x, state.player_y, state.player_direction, state.last_walked_corner, state.walked_on_paths, state.walked_on_corners, state.completed_rectangles, False, False), self.constants, state, action)
            (points_scored_player_step, player_x, player_y, player_direction, last_walked_corner, walked_on_paths, walked_on_corners, completed_rectangles, corners_completed, next_level) = player_state

            def enemies_move_branch():
                rng, enemies_rng = jax.random.split(state.random_key)
                enemy_positions, enemy_directions = enemies_step(self.constants, state, enemies_rng)
                return rng, enemy_positions, enemy_directions

            def enemies_stay_branch():
                return state.random_key, state.enemy_positions, state.enemy_directions

            random_key, enemy_positions, enemy_directions = jax.lax.cond(get_enemy_speed(state.frame_counter, state.level), enemies_move_branch, enemies_stay_branch)

            # CHICKEN MODE-handling
            enemy_types, chicken_counter, chicken_mode_active, jump_counter = chicken_mode(self.constants, state.level, corners_completed, state.enemy_types, state.chicken_counter, state.jump_counter)
            # jax.lax.cond(jnp.any(enemy_types == self.constants.CHICKEN), lambda: jax.debug.print("Chickens activated, {}", chicken_counter), lambda: None)

            # Check for collisions with enemies
            collisions = check_for_collisions(self.constants, player_x, player_y, enemy_positions, enemy_types)
            # collisions = jnp.zeros_like(collisions, dtype=jnp.bool_) # FOR DEBUGGING OTHER THINGS: remove collisions
            # jax.debug.print("Collisions: {collisions}", collisions=collisions)
            (enemy_types, points_scored_enemy_collision, lost_live) = jax.lax.cond(
                jnp.any(collisions),
                lambda: handle_collisions(self.constants, enemy_types, collisions),
                lambda: (enemy_types, 0, False)
            )

            # Jump-handling
            enemy_types, jump_counter, times_jumped = jump(self.constants, state.level, state.frame_counter, action, jump_counter, chicken_mode_active, state.times_jumped, enemy_types)

            # Reset positions if a life is lost 
            reset_key = jax.lax.cond(self.constants.DETERMINISTIC_MODE, lambda: self.constants.DETERMINISTIC_KEY, lambda: random_key)
            random_key, enemy_positions, enemy_directions, player_x, player_y, player_direction, lives, freeze_counter, times_jumped = jax.lax.cond(lost_live,
                lambda: (reset_key, self.constants.INITIAL_ENEMY_POSITIONS, self.constants.INITIAL_ENEMY_DIRECTIONS, self.constants.INITIAL_PLAYER_POSITION[0], self.constants.INITIAL_PLAYER_POSITION[1], self.constants.INITIAL_PLAYER_DIRECTION, state.lives-1, self.constants.FREEZE_DURATION, 0),  # Reset enemy & player positions and directions, decrement lives
                lambda: (random_key,enemy_positions, enemy_directions, player_x, player_y, player_direction, state.lives, state.freeze_counter, times_jumped))  # Keep the current enemy positions and directions

            # Update the level if all edges are completed
            random_key, level, lives, enemy_positions, enemy_directions, enemy_types, freeze_counter, walked_on_paths, walked_on_corners, completed_rectangles, player_x, player_y, player_direction, chicken_counter, jump_counter, times_jumped = jax.lax.cond(next_level, activate_next_level, lambda constants, reset_key, level, lives, enemy_positions, enemy_directions, enemy_types: (random_key, level, lives, enemy_positions, enemy_directions, enemy_types, freeze_counter, walked_on_paths, walked_on_corners, completed_rectangles, player_x, player_y, player_direction, chicken_counter, jump_counter, times_jumped), self.constants, reset_key, state.level, lives, enemy_positions, enemy_directions, enemy_types)

            new_state = AmidarState(
                frame_counter=state.frame_counter + 1,  # Increment the frame counter
                random_key=random_key,
                freeze_counter=freeze_counter,
                level=level,
                score=state.score + points_scored_player_step + points_scored_enemy_collision, # Could change in multiple functions, so it is not calculated in the function based on the previous state
                lives=lives,
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                last_walked_corner=last_walked_corner,
                walked_on_paths=walked_on_paths,
                walked_on_corners=walked_on_corners,
                completed_rectangles=completed_rectangles,
                enemy_positions=enemy_positions,
                enemy_directions=enemy_directions,
                enemy_types=enemy_types,
                chicken_counter=chicken_counter,
                jump_counter=jump_counter,
                times_jumped=times_jumped,
            )
            
            return new_state
        
        def freeze_game():
            new_state = state._replace(freeze_counter=state.freeze_counter - 1) # Decrement the freeze counter
            new_state = new_state._replace(frame_counter=state.frame_counter + 1)  # Update the frame counter to ensure it changes with each step
            return new_state
        
        # Check if the game is frozen
        is_frozen = state.freeze_counter > 0
        # jax.debug.print("Freeze counter: {freeze_counter}", freeze_counter=state.freeze_counter)
        new_state= jax.lax.cond(is_frozen, freeze_game, take_step)


        observation = self._get_observation(new_state)
        # jax.debug.print("observation{}", observation)

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(previous_state, new_state)
        all_rewards = self._get_all_rewards(previous_state, new_state)
        info = self._get_info(new_state, all_rewards)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AmidarState):
        # create player
        player_gorilla = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.constants.PLAYER_SIZE[0]),
            height=jnp.array(self.constants.PLAYER_SIZE[1]),
            active=state.level % 2 == 1,
        )

        player_paint_roller = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.constants.PLAYER_SIZE[0]),
            height=jnp.array(self.constants.PLAYER_SIZE[1]),
            active=state.level % 2 == 0,
        )

        # Define a function to convert enemy positions to entity format
        def convert_to_entity(pos, size, active):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                active,  # active flag
            ])

        # Apply conversion to each type of entity using vmap

        true = jnp.full_like(state.enemy_types, True)
        false = jnp.full_like(state.enemy_types, False)

        # shadows
        is_shadow = jnp.where(state.enemy_types == self.constants.SHADOW, true, false)
        shadows = jax.vmap(convert_to_entity, in_axes=(0, None, 0))(state.enemy_positions, self.constants.ENEMY_SIZE, is_shadow)

        # warriors
        is_warrior = jnp.where(state.enemy_types == self.constants.WARRIOR, true, false)
        warriors = jax.vmap(convert_to_entity, in_axes=(0, None, 0))(state.enemy_positions, self.constants.ENEMY_SIZE, is_warrior)
        
        # pigs
        is_pig = jnp.where(state.enemy_types == self.constants.PIG, true, false)
        pigs = jax.vmap(convert_to_entity, in_axes=(0, None, 0))(state.enemy_positions, self.constants.ENEMY_SIZE, is_pig)

        # chickens
        is_chicken = jnp.where(state.enemy_types == self.constants.CHICKEN, true, false)
        chickens = jax.vmap(convert_to_entity, in_axes=(0, None, 0))(state.enemy_positions, self.constants.CHICKEN_SIZE, is_chicken)

        def make_path_edge_entity_horizontal(edge):
            start = edge[0]
            end = edge[1]
            return jnp.array([
                start[0],  # x position
                start[1],  # y position
                end[0] - start[0],  # width
                self.constants.PATH_THICKNESS_HORIZONTAL,  # height
                jnp.array(1),  # Path is always active
            ])

        def make_path_edge_entity_vertical(edge):
            start = edge[0]
            end = edge[1]
            return jnp.array([
                start[0],  # x position
                start[1],  # y position
                self.constants.PATH_THICKNESS_VERTICAL,  # width
                end[1] - start[1],  # height
                jnp.array(1),  # Path is always active
            ])
        
        horizontal_edges = jax.vmap(make_path_edge_entity_horizontal, in_axes=0)(self.constants.HORIZONTAL_PATH_EDGES)
        vertical_edges = jax.vmap(make_path_edge_entity_vertical, in_axes=0)(self.constants.VERTICAL_PATH_EDGES)

        paths = jnp.concatenate([horizontal_edges, vertical_edges], axis=0)
        walked_on_paths = paths.at[:, 4].set(state.walked_on_paths.astype(paths.dtype))

        def make_completed_rectangle_entity(rectangle_bound, completed):
            return jnp.array([
                rectangle_bound[0],  # x position
                rectangle_bound[1],  # y position
                rectangle_bound[2] - rectangle_bound[0],  # width
                rectangle_bound[3] - rectangle_bound[1],  # height
                completed,  # active flag
            ])

        completed_rectangles = jax.vmap(make_completed_rectangle_entity, in_axes=(0, 0))(self.constants.RECTANGLE_BOUNDS, state.completed_rectangles)

        return AmidarObservation(
            player_gorilla=player_gorilla,
            player_paint_roller=player_paint_roller,
            shadows=shadows,
            warriors=warriors,
            pigs=pigs,
            chickens=chickens,
            lives=state.lives,
            paths=paths,
            walked_on_paths=walked_on_paths,
            completed_rectangles=completed_rectangles
        )
        
    
    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

def load_sprites():
    """Load all sprites required for Amidar rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites

    DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/amidar/score/{}.npy"))

    player_ghost = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/player_ghost.npy"))

    player_paint_roller = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/player_paint_roller.npy"))

    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/background.npy"), transpose=True)

    life = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/life.npy"))

    warrior = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/enemy/warrior.npy"))

    pig = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/enemy/pig.npy"))

    chicken = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/enemy/chicken.npy"))

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_LIFE = jnp.expand_dims(life, axis=0)
    SPRITE_PLAYER_GHOST = jnp.expand_dims(player_ghost, axis=0)
    SPRITE_PLAYER_PAINT_ROLLER = jnp.expand_dims(player_paint_roller, axis=0)
    SPRITE_WARRIOR = jnp.expand_dims(warrior, axis=0)
    SPRITE_PIG = jnp.expand_dims(pig, axis=0)
    SPRITE_CHICKEN = jnp.expand_dims(chicken, axis=0)

    return (
        SPRITE_BG,
        DIGITS,
        SPRITE_LIFE,
        SPRITE_PLAYER_GHOST,
        SPRITE_PLAYER_PAINT_ROLLER,
        SPRITE_WARRIOR,
        SPRITE_PIG,
        SPRITE_CHICKEN,
    )


class AmidarRenderer(JAXGameRenderer):
    """JAX-based Amidar game renderer, optimized with JIT compilation."""

    def __init__(self, constants: AmidarConstants = None):
        self.constants = constants or AmidarConstants()
        (
            self.SPRITE_BG,
            self.DIGITS,
            self.SPRITE_LIFE,
            self.SPRITE_PLAYER_GHOST,
            self.SPRITE_PLAYER_PAINT_ROLLER,
            self.SPRITE_WARRIOR,
            self.SPRITE_PIG,
            self.SPRITE_CHICKEN,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A AmidarState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen

        raster = jnp.zeros((self.constants.HEIGHT, self.constants.WIDTH, 3))
        
        empty_mask = jnp.zeros((self.constants.WIDTH, self.constants.HEIGHT), dtype=jnp.uint8)
        empty_raster = jnp.zeros_like(raster, dtype=jnp.uint8)

        # Render background - (0, 0) is top-left corner
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render paths
        path_sprite = jax.lax.cond(state.level%2 == 1, lambda: self.constants.PATH_SPRITE_BROWN, lambda: self.constants.PATH_SPRITE_GREEN)
        raster = aj.render_at(raster, 0, 0, path_sprite)

        # Render walked on paths
        walked_on_paths_horizontal = state.walked_on_paths[0:jnp.shape(self.constants.HORIZONTAL_PATH_EDGES)[0]]
        walked_on_paths_vertical = state.walked_on_paths[jnp.shape(self.constants.HORIZONTAL_PATH_EDGES)[0]:]
        _, walked_on_rendering_mask = generate_path_mask(self.constants.WIDTH, self.constants.HEIGHT, self.constants.PATH_THICKNESS_HORIZONTAL, self.constants.PATH_THICKNESS_VERTICAL, self.constants.HORIZONTAL_PATH_EDGES, self.constants.VERTICAL_PATH_EDGES, self.constants.PATH_CORNERS, horizontal_cond=walked_on_paths_horizontal, vertical_cond=walked_on_paths_vertical, corner_cond=state.walked_on_corners)
        walked_on_paths_sprite = jnp.where(walked_on_rendering_mask[:, :, None] == 1, self.constants.WALKED_ON_PATTERN, jnp.full((self.constants.HEIGHT, self.constants.WIDTH, 4), 0, dtype=jnp.uint8))
        raster = aj.render_at(raster, 0, 0, walked_on_paths_sprite)

        # Render completed rectangles
        # Still computed in the (HEIGHT, WIDTH) format and transposed later since the code uses (x, y) coordinates
        # Create coordinate grids for vectorized operations
        x_coords, y_coords = jnp.meshgrid(jnp.arange(self.constants.WIDTH), jnp.arange(self.constants.HEIGHT), indexing='ij')
        def create_rectangle_mask(rectangle_bounds, is_completed):
            rect_min_x, rect_min_y, rect_max_x, rect_max_y = rectangle_bounds
            def create_mask():
                in_rect_x = jnp.logical_and(x_coords >= rect_min_x, x_coords <= rect_max_x)
                in_rect_y = jnp.logical_and(y_coords >= rect_min_y, y_coords <= rect_max_y)
                rect_mask = jnp.logical_and(in_rect_x, in_rect_y).astype(jnp.uint8)
                return rect_mask
            return jax.lax.cond(is_completed, create_mask, lambda: empty_mask)
        rectangle_masks = jax.vmap(create_rectangle_mask, in_axes=(0, 0))(self.constants.RECTANGLE_BOUNDS, state.completed_rectangles)
        completed_rectangles_mask = jnp.any(rectangle_masks, axis=0).astype(jnp.uint8)
        completed_rectangles_mask = jnp.transpose(completed_rectangles_mask, (1, 0))  # Transpose to match the HWC format for rendering

        completed_rectangles_sprite = jnp.where(completed_rectangles_mask[:, :, None] == 1, self.constants.WALKED_ON_PATTERN, jnp.full((self.constants.HEIGHT, self.constants.WIDTH, 4), 0, dtype=jnp.uint8))
        raster = aj.render_at(raster, 0, 0, completed_rectangles_sprite)

        # Render score
        max_digits = 8  # Maximum number of digits to render
        score_array = aj.int_to_digits(state.score, max_digits=max_digits)
        # convert the score to a list of digits
        number_of_digits = (jnp.log10(state.score)+1).astype(jnp.int32)
        number_of_digits = jnp.maximum(number_of_digits, 1)  # Ensure at least one digit is rendered

        def get_digit_sprite(i):
            
            def get_digit_sprite():
                digit_index_in_array = 8 - number_of_digits + i
                digit_value = score_array[digit_index_in_array]
                sprite_to_render = self.DIGITS[digit_value] # Gets (W, H, C) sprite
                return sprite_to_render

            # if there is no digit, return an empty sprite
            sprite_to_render = jax.lax.cond(i >= number_of_digits, lambda: jnp.zeros((7, 7, 4), dtype=jnp.uint8), get_digit_sprite)
            return sprite_to_render

        digit_sprites = jax.vmap(get_digit_sprite)(jnp.arange(max_digits))  # Render all digits in parallel
        x_positions = jnp.arange(0, max_digits * 8, step=8) + 105-(number_of_digits * 8)  # Calculate x positions for each digit
        digit_rasters = jax.vmap(aj.render_at, in_axes=(None, 0, None, 0))(empty_raster, x_positions, 176, digit_sprites)
        digits_raster = jnp.sum(digit_rasters, axis=0)  # Combine all digit rasters into one, adding works because irrelevant values are zero
        raster = jnp.add(raster, digits_raster)  # Combine the raster with the digits raster

        # Render lives
        max_lives_rendered = 3
        lives = state.lives
        life_sprite = aj.get_sprite_frame(self.SPRITE_LIFE, 0)  # Get the life sprite frame
        life_sprites = jax.vmap(lambda i: jax.lax.cond(i < lives, lambda: life_sprite, lambda: jnp.zeros_like(life_sprite, dtype=jnp.uint8)))(jnp.arange(max_lives_rendered))
        x_positions = 148 - jnp.arange(0, max_lives_rendered * 16, step=16)  # Calculate x positions for each life
        life_rasters = jax.vmap(aj.render_at, in_axes=(None, 0, None, 0))(empty_raster, x_positions, 175, life_sprites)
        lives_raster = jnp.sum(life_rasters, axis=0)  # Combine all life rasters into one
        raster = jnp.add(raster, lives_raster)  # Combine the raster with the lives raster

        # Render enemies
        frame_warrior = aj.get_sprite_frame(self.SPRITE_WARRIOR, 0)
        frame_pig = aj.get_sprite_frame(self.SPRITE_PIG, 0)
        frame_chicken = aj.get_sprite_frame(self.SPRITE_CHICKEN, 0)
        frame_shadow = jnp.broadcast_to(jnp.array([0, 0, 0, 255]), frame_warrior.shape).at[5:].set(0).at[:, :3].set(0)
        # jax.debug.print("w: {}, c: {}", frame_warrior.shape, frame_chicken.shape)
        frames_enemies = jnp.array([frame_shadow, frame_warrior, frame_pig, frame_chicken, jnp.zeros_like(frame_warrior)])

        # Use scan to accumulate the raster updates
        def scan_render_enemy(raster, enemy_data):
            enemy_pos, enemy_type = enemy_data
            frame_enemy = frames_enemies[enemy_type]  # Get the correct frame for the enemy type
            new_raster = aj.render_at(raster, enemy_pos[0]+self.constants.ENEMY_SPRITE_OFFSET[0], enemy_pos[1]+self.constants.ENEMY_SPRITE_OFFSET[1], frame_enemy)
            return new_raster, None

        raster, _ = jax.lax.scan(scan_render_enemy, raster, (state.enemy_positions, state.enemy_types))
        
        # Render player
        sprite_player = jax.lax.cond(state.level%2 == 1, lambda: self.SPRITE_PLAYER_GHOST, lambda: self.SPRITE_PLAYER_PAINT_ROLLER)
        frame_player = aj.get_sprite_frame(sprite_player, 0)
        raster = aj.render_at(raster, state.player_x+self.constants.PLAYER_SPRITE_OFFSET[0], state.player_y+self.constants.PLAYER_SPRITE_OFFSET[1], frame_player)

        ###### For DEBUGGING #######
        # loops are fine for debugging 

        # # Render path edges
        # def render_path(i, raster):
        #     path = jnp.array(self.constants.PATH_EDGES[i])
        #     raster = render_line(raster, path, (255, 0, 0))
        #     return raster
        # raster = jax.lax.fori_loop(0, jnp.shape(self.constants.PATH_EDGES)[0], render_path, raster)

        # # Render path mask
        # transposed_path_mask = jnp.transpose(self.constants.PATH_MASK, (1, 0))  # Transpose to match the HWC format for rendering
        # all_white = jnp.full_like(raster, 255, dtype=jnp.uint8)
        # raster = jnp.where(transposed_path_mask[:, :, None] == 1, all_white, raster)

        # # Render rendering path mask
        # all_white = jnp.full_like(raster, 255, dtype=jnp.uint8)
        # raster = jnp.where(self.constants.RENDERING_PATH_MASK[:, :, None] == 1, all_white, raster)

        # # Render corner rectangles
        # edges_in_corner_rectangle = jnp.any(self.constants.RECTANGLES[self.constants.CORNER_RECTANGLES], axis=0)
        # def render_corner_rectangle_edges(i, raster):
        #     in_corner = edges_in_corner_rectangle[i] == 1   
        #     raster = jax.lax.cond(
        #         in_corner,
        #         lambda raster: render_line(raster, self.constants.PATH_EDGES[i], (0, 255, 0)),  # Render in green if part of a corner rectangle
        #         lambda raster: raster,  # Otherwise, do nothing
        #         raster
        #     )
        #     return raster
        # raster = jax.lax.fori_loop(0, jnp.shape(self.constants.PATH_EDGES)[0], render_corner_rectangle_edges, raster)


        # # Render completed rectangles mask
        # all_white = jnp.full_like(raster, 255, dtype=jnp.uint8)
        # raster = jnp.where(completed_rectangles_mask[:, :, None] == 1, all_white, raster)

        # # Render path pattern
        # raster = aj.render_at(raster, 0, 0, self.constants.PATH_PATTERN)

        # # Render walked on pattern
        # raster = aj.render_at(raster, 0, 0, self.constants.WALKED_ON_PATTERN)

        # # Render walked on paths
        # def render_walked_paths(i, raster):
        #     # Check if the path edge has been walked on
        #     walked_on = state.walked_on_paths[i] == 1   
            
        #     raster = jax.lax.cond(
        #         walked_on,
        #         lambda raster: render_line(raster, self.constants.PATH_EDGES[i], (0, 255, 0)),  # Render in green if walked on
        #         lambda raster: raster,  # Otherwise, do nothing
        #         raster
        #     )

        #     return raster

        # raster = jax.lax.fori_loop(0, jnp.shape(self.constants.PATH_EDGES)[0], render_walked_paths, raster)

        return jnp.array(raster, dtype=jnp.uint8)  # Ensure the raster is returned as a JAX array with uint8 type

@jax.jit
def render_line(raster, coords, color):
    """Renders a line on the raster from coords [[x1, y1], [x2, y2]] with the given color.
    Coordinates are given as (w, h) i.e., (x, y), while raster is indexed as (h, w, c).

    Args:
        raster: JAX array of shape (Height, Width, Channels) for the target image.
        coords: JAX array of shape (2, 2), where coords[0] = [x1, y1] and coords[1] = [x2, y2].
        color: RGB or RGBA tuple/list/array for the line color.

    Returns:
        Updated raster with the line rendered.
    """
    color = jnp.asarray(color)
    coords = jnp.asarray(coords, dtype=jnp.int32)

    if color.shape[0] not in (3, 4):
        raise ValueError("Color must be an RGB or RGBA array.")

    # Match raster dtype and clamp channels to raster channels (use static Python int to avoid dynamic slice)
    channels = min(color.shape[0], raster.shape[2])
    color = color[:channels].astype(raster.dtype)

    x1, y1 = coords[0]
    x2, y2 = coords[1]

    # Bresenham parameters
    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)
    sx = jnp.where(x1 < x2, 1, -1)
    sy = jnp.where(y1 < y2, 1, -1)
    err = dx - dy

    # Steps: cover longest axis
    num_steps = jnp.maximum(dx, dy) + 1

    raster_h, raster_w, _ = raster.shape

    def loop_body(i, carry):
        raster, x, y, err = carry

        # Bounds check for (y, x) since raster is (h, w, c)
        in_bounds = jnp.logical_and(
            jnp.logical_and(0 <= x, x < raster_w),
            jnp.logical_and(0 <= y, y < raster_h),
        )

        def set_pixel(r):
            return r.at[y, x, :channels].set(color)

        raster = jax.lax.cond(in_bounds, set_pixel, lambda r: r, operand=raster)

        # Update Bresenham
        e2 = 2 * err
        err = jnp.where(e2 > -dy, err - dy, err)
        x = jnp.where(e2 > -dy, x + sx, x)
        err = jnp.where(e2 < dx, err + dx, err)
        y = jnp.where(e2 < dx, y + sy, y)

        return raster, x, y, err

    raster, _, _, _ = jax.lax.fori_loop(0, num_steps, loop_body, (raster, x1, y1, err))

    return raster

if __name__ == "__main__":
    print("please use the play script to run the game")