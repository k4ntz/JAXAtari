# next TODO's (game features): jumping, chicken mode, Level 2
# TODO update observation, freeze for about 250 frames, vmap more, random blinking, update spaces, possibly make generating/changing the maze easier?, update info

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

# Functions to precompute some constants so they only need to be calculated once

def calculate_rectangles(HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES, PATH_EDGES, PATH_CORNERS):
    """not jited since it is only ran once; assumes the vertical edges are top to bottom and horizontal edges are left to right"""

    def convert_vertical_index(index):
        """Converts an index in VERTICAL_PATH_EDGES to the corresponding index in PATH_EDGES."""
        return index + HORIZONTAL_PATH_EDGES.shape[0]
    
    def add_edge(rectangle, edge_index):
        """Adds an edge to the rectangle."""
        rectangle = rectangle.at[edge_index].set(1)
        return rectangle
    
    no_edges = jnp.zeros(PATH_EDGES.shape[0], dtype=jnp.int32)
    
    def left_and_check_down(rectangle, corner):
        """Checks if there is a horizontal edge going left from the corner and adds it to the rectangle,
        then checks if there is a vertical edge going down from the new corner (in this case the rectangle should be complete)."""
        if jnp.any(jnp.apply_along_axis(jnp.all, 1, (HORIZONTAL_PATH_EDGES[:, 1] == corner))):
            edge_index = jnp.where(jnp.apply_along_axis(jnp.all, 1, (HORIZONTAL_PATH_EDGES[:, 1] == corner)), size=1)[0][0]
            rectangle = add_edge(rectangle, edge_index)
            new_corner = PATH_EDGES[edge_index, 0]

            if not jnp.any(jnp.apply_along_axis(jnp.all, 1, (VERTICAL_PATH_EDGES[:, 0] == new_corner))): # if there is NOT a vertical edge that goes down from there
                rectangle, new_corner = left_and_check_down(rectangle, new_corner) # find another edge that goes left
            
            return rectangle, new_corner
        else: 
            return no_edges, None
    
    def up_and_check_left(rectangle, corner):
        """Checks if there is a vertical edge going up from the corner and adds it to the rectangle,
        then checks if there is a horizontal edge going left from the new corner."""
        if jnp.any(jnp.apply_along_axis(jnp.all, 1, (VERTICAL_PATH_EDGES[:, 1] == corner))): # if there is an edge that goes up
            edge_index = convert_vertical_index(jnp.where(jnp.apply_along_axis(jnp.all, 1, (VERTICAL_PATH_EDGES[:, 1] == corner)), size=1)[0][0]) # get the index of the edge that goes up
            rectangle = add_edge(rectangle, edge_index)  # Add the vertical edge going up
            new_corner = PATH_EDGES[edge_index, 0]  # This is the new corner after going up

            if jnp.any(jnp.apply_along_axis(jnp.all, 1, (HORIZONTAL_PATH_EDGES[:, 1] == new_corner))): # if there is a horizontal edge that goes left from there 
                rectangle, new_corner = left_and_check_down(rectangle, new_corner)
            else:
                rectangle, new_corner = up_and_check_left(rectangle, new_corner) # find another edge that goes up

            return rectangle, new_corner
        else: 
            return no_edges, None

    def right_and_check_up(rectangle, corner):
        """Checks if there is a horizontal edge going right from the corner and adds it to the rectangle,
        then checks if there is a vertical edge going up from the new corner."""
        if jnp.any(jnp.apply_along_axis(jnp.all, 1, (HORIZONTAL_PATH_EDGES[:, 0] == corner))):
            edge_index = jnp.where(jnp.apply_along_axis(jnp.all, 1, (HORIZONTAL_PATH_EDGES[:, 0] == corner)), size=1)[0][0]
            rectangle = add_edge(rectangle, edge_index)
            new_corner = PATH_EDGES[edge_index, 1]

            if jnp.any(jnp.apply_along_axis(jnp.all, 1, (VERTICAL_PATH_EDGES[:, 1] == new_corner))): # if there is a vertical edge that goes up from there 
                rectangle, new_corner = up_and_check_left(rectangle, new_corner)
            else:
                rectangle, new_corner = right_and_check_up(rectangle, new_corner) #find another edge that goes right
            
            return rectangle, new_corner
        else: 
            return no_edges, None

    def down_and_check_right(rectangle, corner):
        """Checks if there is a vertical edge going down from the corner and adds it to the rectangle,
        then checks if there is a horizontal edge going right from the new corner."""
        if jnp.any(jnp.apply_along_axis(jnp.all, 1, (VERTICAL_PATH_EDGES[:, 0] == corner))): # if there is an edge that goes down
            edge_index = convert_vertical_index(jnp.where(jnp.apply_along_axis(jnp.all, 1, (VERTICAL_PATH_EDGES[:, 0] == corner)), size=1)[0][0]) # get the index of the edge that goes down
            rectangle = add_edge(rectangle, edge_index)  # Add the vertical edge going down
            new_corner = PATH_EDGES[edge_index, 1]  # This is the new corner after going down

            if jnp.any(jnp.apply_along_axis(jnp.all, 1, (HORIZONTAL_PATH_EDGES[:, 0] == new_corner))): # if there is a horizontal edge that goes right from there
                rectangle, new_corner = right_and_check_up(rectangle, new_corner)
            else:
                rectangle, new_corner = down_and_check_right(rectangle, new_corner) # find another edge that goes down
            
            return rectangle, new_corner
        else: 
            return no_edges, None

    rectangles = []
    for corner in PATH_CORNERS:
        rectangle = jnp.zeros(PATH_EDGES.shape[0], dtype=jnp.int32)
        
        rectangle, new_corner = down_and_check_right(rectangle, corner)

        if jnp.array_equal(rectangle, no_edges) or not jnp.array_equal(corner, new_corner):
            continue
        else:
            # If the rectangle is not empty and we arrived back at the starting corner, add it to the list
            rectangles.append(rectangle)

    return rectangles

@jax.jit
def precompute_rectangle_corners(rectangle, PATH_EDGES):
    """Gets the top-left and bottom-right corners of a rectangle."""
    def calculate():
        # Get only the edges that belong to this rectangle
        rectangle_edges = jnp.where(rectangle[:, None, None], PATH_EDGES, jnp.full(PATH_EDGES.shape, jnp.inf, dtype=jnp.float32))
        
        # Flatten the coordinates to make min/max operations easier
        all_x_coords = rectangle_edges[:, :, 0].flatten()
        all_y_coords = rectangle_edges[:, :, 1].flatten()
        
        # Instead of boolean indexing, use conditional operations
        # Find min/max while ignoring inf values
        min_x = jnp.min(jnp.where(all_x_coords != jnp.inf, all_x_coords, jnp.inf)).astype(jnp.int32)
        min_y = jnp.min(jnp.where(all_y_coords != jnp.inf, all_y_coords, jnp.inf)).astype(jnp.int32)
        max_x = jnp.max(jnp.where(all_x_coords != jnp.inf, all_x_coords, -jnp.inf)).astype(jnp.int32)
        max_y = jnp.max(jnp.where(all_y_coords != jnp.inf, all_y_coords, -jnp.inf)).astype(jnp.int32)
        
        return jnp.stack((min_x, min_y, max_x, max_y))
    return jax.lax.cond(jnp.any(rectangle), calculate, lambda: jnp.zeros(4, dtype=jnp.int32))

@partial(jax.jit, static_argnames=['WIDTH', 'HEIGHT'])
def generate_path_mask(WIDTH, HEIGHT, horizontal_edges, vertical_edges, horizontal_cond, vertical_cond):
    """Generates a mask for the path edges. Conditions are used so this function can also be used to render only the walked on paths."""
    # Create an empty mask 
    mask = jnp.zeros((WIDTH, HEIGHT), dtype=jnp.int32)
    rendering_mask = jnp.zeros((WIDTH, HEIGHT), dtype=jnp.int32)

    def add_horizontal_edge(i, carry):
        mask, rendering_mask = carry
        start, end = horizontal_edges[i]

        x1, y1 = start
        x2, y2 = end

        def loop(x, carry):
            mask, rendering_mask, y = carry
            mask = mask.at[x, y].set(1)
            rendering_mask = rendering_mask.at[x, y].set(1)
            rendering_mask = rendering_mask.at[x, y+1].set(1)
            rendering_mask = rendering_mask.at[x, y+2].set(1)
            rendering_mask = rendering_mask.at[x, y+3].set(1)
            rendering_mask = rendering_mask.at[x, y+4].set(1)
            return (mask, rendering_mask, y)

        mask, rendering_mask, _ = jax.lax.cond(horizontal_cond[i], lambda: jax.lax.fori_loop(x1, x2 + 1, loop, (mask, rendering_mask, y1)), lambda: (mask, rendering_mask, y1))

        # add a bit to the rendering mask to make sure that even at corners the path is visible        
        def add_corners_for_rendering(x, carry):
            rendering_mask, y = carry
            rendering_mask = rendering_mask.at[x, y].set(1)
            rendering_mask = rendering_mask.at[x, y+1].set(1)
            rendering_mask = rendering_mask.at[x, y+2].set(1)
            rendering_mask = rendering_mask.at[x, y+3].set(1)
            rendering_mask = rendering_mask.at[x, y+4].set(1)
            return (rendering_mask, y)

        rendering_mask, _ = jax.lax.cond(horizontal_cond[i], lambda: jax.lax.fori_loop(x2, x2 + 4, add_corners_for_rendering, (rendering_mask, y1)), lambda: (rendering_mask, y1))

        return mask, rendering_mask

    def add_vertical_edge(i, carry):
        mask, rendering_mask = carry
        start, end = vertical_edges[i]

        x1, y1 = start
        x2, y2 = end

        def loop(y, carry):
            mask, rendering_mask, x = carry
            mask = mask.at[x, y].set(1)
            rendering_mask = rendering_mask.at[x, y].set(1)
            rendering_mask = rendering_mask.at[x+1, y].set(1)
            rendering_mask = rendering_mask.at[x+2, y].set(1)
            rendering_mask = rendering_mask.at[x+3, y].set(1)
            return (mask, rendering_mask, x)

        mask, rendering_mask, _ = jax.lax.cond(vertical_cond[i], lambda: jax.lax.fori_loop(y1, y2 + 1, loop, (mask, rendering_mask, x1)), lambda: (mask, rendering_mask, x1))
        return mask, rendering_mask

    mask, rendering_mask = jax.lax.fori_loop(0, jnp.shape(horizontal_edges)[0], add_horizontal_edge, (mask, rendering_mask))
    mask, rendering_mask = jax.lax.fori_loop(0, jnp.shape(vertical_edges)[0], add_vertical_edge, (mask, rendering_mask))

    # transpose to match the HWC format for rendering
    rendering_mask = jnp.transpose(rendering_mask, (1, 0))

    return mask, rendering_mask

@partial(jax.jit, static_argnames=['WIDTH', 'HEIGHT'])
def generate_path_pattern(WIDTH, HEIGHT, PATH_COLOR, WALKED_ON_COLOR):
    """Generates a path pattern for rendering.
    Returns a JAX array of shape (WIDTH, HEIGHT) with the path pattern."""
    # Create an empty mask
    path_pattern = jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8)
    walked_on_pattern = jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8)

    # put the indices in a seperate array to be able to use vmap
    ii, jj = jnp.meshgrid(jnp.arange(HEIGHT), jnp.arange(WIDTH), indexing='ij')
    indices = jnp.stack((ii, jj), axis=-1)  # shape (HEIGHT, WIDTH, 2)

    def set_for_column(path_column, walked_on_column, indices):

        def set_color(path_value, walked_on_value, index):
            x, y = index

            path_value, walked_on_value, index = jax.lax.cond(jnp.logical_or(jnp.logical_or(x % 5 == 0, x % 5 == 2), x % 5 == 3),
                lambda: (PATH_COLOR, walked_on_value, index),
                lambda: (path_value, WALKED_ON_COLOR, index)
            )
            return path_value, walked_on_value, index

        path_column, walked_on_column, index = jax.vmap(set_color, in_axes=0)(path_column, walked_on_column, indices)
        return path_column, walked_on_column, index

    path_pattern, walked_on_pattern, _ = jax.vmap(set_for_column, in_axes=0)(path_pattern, walked_on_pattern, indices)

    return path_pattern, walked_on_pattern


class AmidarConstants(NamedTuple):
    """Constants for the Amidar game. Some constants are precomputed from others to avoid recomputation."""
    # General
    WIDTH: int = 160
    HEIGHT: int = 210
    RANDOM_KEY_SEED: int = 0
    INITIAL_LIVES: int = 3
    FREEZE_DURATION: int = 256  # Duration for which the game is frozen in the beginning and after being hit by an enemy

    # Rendering
    PATH_COLOR = jnp.array([162, 98, 33, 255], dtype=jnp.uint8)  # Brown color for the path
    WALKED_ON_COLOR = jnp.array([104, 72, 198, 255], dtype=jnp.uint8)  # Purple color for the walked on paths

    # Points
    PIXELS_PER_POINT_HORIZONTAL: int = 3 # Values to calculate how many points an Edge is worth based on how long it is
    PIXELS_PER_POINT_VERTICAL: int = 30 # Each vertical edge is worth 1 point, since they are 30 pixels long
    BONUS_POINTS_PER_RECTANGLE: int = 48 # Bonus points for completing a rectangle

    # Player
    PLAYER_SIZE: tuple[int, int] = (7, 7)  # Object sizes (width, height)
    ENEMY_SIZE: tuple[int, int] = (7, 7)  # Object sizes (width, height)
    PLAYER_SPRITE_OFFSET: tuple[int, int] = (-1, 0) # Offset for the player sprite in relation to the position in the code (because the top left corner of the player sprite is of the path to the left)
    INITIAL_PLAYER_POSITION: chex.Array = jnp.array([140, 88])
    INITIAL_PLAYER_DIRECTION: chex.Array = jnp.array(0)
    PLAYER_STARTING_PATH: int = 85  # The path edge the player starts on, this is the index in PATH_EDGES

    # Enemies
    INITIAL_ENEMY_POSITIONS: chex.Array = jnp.array(
        [   [16, 14],  # Enemy 1
            [16, 14],  # Enemy 2
            [44, 14],  # Enemy 3
            [16, 137],  # Enemy 4
            [16, 164],  # Enemy 5
            [52, 164],  # Enemy 6
        ])
    INITIAL_ENEMY_DIRECTIONS: chex.Array = jnp.array([3, 3, 3, 3, 3, 3])  # All enemies start moving right
    INITIAL_ENEMY_TYPES: chex.Array = jnp.array([1, 1, 1, 1, 1, 1])
    ENEMY_SPRITE_OFFSET: tuple[int, int] = (-1, 0) # Offset for the enemy sprite in relation to the position in the code (because the top left corner of the enemy sprite is of the path to the left)
    
    # Path Structure
    PATH_CORNERS: chex.Array = jnp.array([[16, 14], [40, 14], [56, 14], [72, 14], [84, 14], [100, 14], [116, 14], [140, 14], 
                            [16, 44], [40, 44], [56, 44], [72, 44], [84, 44], [100, 44], [116, 44], [140, 44], 
                            [32, 44], [52, 44], [64, 44], [92, 44], [104, 44], [124, 44], [16, 74], [32, 74], 
                            [52, 74], [64, 74], [92, 74], [104, 74], [124, 74], [140, 74], [28, 74], [60, 74], 
                            [96, 74], [128, 74], [16, 104], [28, 104], [60, 104], [96, 104], [128, 104], [140, 104], 
                            [36, 104], [72, 104], [84, 104], [120, 104], [16, 134], [36, 134], [72, 134], [84, 134], 
                            [120, 134], [140, 134], [40, 134], [64, 134], [92, 134], [116, 134], [16, 164], [40, 164], 
                            [64, 164], [92, 164], [116, 164], [140, 164]]
    ) # Coordinates where at least two path edges meet
    HORIZONTAL_PATH_EDGES: chex.Array = jnp.array([[[16, 14], [40, 14]], [[40, 14], [56, 14]], [[56, 14], [72, 14]], [[72, 14], [84, 14]], 
                                    [[84, 14], [100, 14]], [[100, 14], [116, 14]], [[116, 14], [140, 14]], [[16, 44], [32, 44]], 
                                    [[32, 44], [40, 44]], [[40, 44], [52, 44]], [[52, 44], [56, 44]], [[56, 44], [64, 44]], 
                                    [[64, 44], [72, 44]], [[72, 44], [84, 44]], [[84, 44], [92, 44]], [[92, 44], [100, 44]], 
                                    [[100, 44], [104, 44]], [[104, 44], [116, 44]], [[116, 44], [124, 44]], [[124, 44], [140, 44]], 
                                    [[16, 74], [28, 74]], [[28, 74], [32, 74]], [[32, 74], [52, 74]], [[52, 74], [60, 74]], 
                                    [[60, 74], [64, 74]], [[64, 74], [92, 74]], [[92, 74], [96, 74]], [[96, 74], [104, 74]], 
                                    [[104, 74], [124, 74]], [[124, 74], [128, 74]], [[128, 74], [140, 74]], [[16, 104], [28, 104]], 
                                    [[28, 104], [36, 104]], [[36, 104], [60, 104]], [[60, 104], [72, 104]], [[72, 104], [84, 104]], 
                                    [[84, 104], [96, 104]], [[96, 104], [120, 104]], [[120, 104], [128, 104]], [[128, 104], [140, 104]], 
                                    [[16, 134], [36, 134]], [[36, 134], [40, 134]], [[40, 134], [64, 134]], [[64, 134], [72, 134]], 
                                    [[72, 134], [84, 134]], [[84, 134], [92, 134]], [[92, 134], [116, 134]], [[116, 134], [120, 134]], 
                                    [[120, 134], [140, 134]], [[16, 164], [40, 164]], [[40, 164], [64, 164]], [[64, 164], [92, 164]], 
                                    [[92, 164], [116, 164]], [[116, 164], [140, 164]]] 
    ) # Tuples of (start, end); coordinates are of the top left corner of the path-corner; x_start must be less than x_end
    VERTICAL_PATH_EDGES: chex.Array = jnp.array([[[16, 14], [16, 44]], [[16, 44], [16, 74]], [[16, 74], [16, 104]], [[16, 104], [16, 134]], 
                                    [[16, 134], [16, 164]], [[28, 74], [28, 104]], [[32, 44], [32, 74]], [[36, 104], [36, 134]], 
                                    [[40, 14], [40, 44]], [[40, 134], [40, 164]], [[52, 44], [52, 74]], [[56, 14], [56, 44]], 
                                    [[60, 74], [60, 104]], [[64, 44], [64, 74]], [[64, 134], [64, 164]], [[72, 14], [72, 44]], 
                                    [[72, 104], [72, 134]], [[84, 14], [84, 44]], [[84, 104], [84, 134]], [[92, 44], [92, 74]], 
                                    [[92, 134], [92, 164]], [[96, 74], [96, 104]], [[100, 14], [100, 44]], [[104, 44], [104, 74]], 
                                    [[116, 14], [116, 44]], [[116, 134], [116, 164]], [[120, 104], [120, 134]], [[124, 44], [124, 74]], 
                                    [[128, 74], [128, 104]], [[140, 14], [140, 44]], [[140, 44], [140, 74]], [[140, 74], [140, 104]], 
                                    [[140, 104], [140, 134]], [[140, 134], [140, 164]]]
    ) # Tuples of (start, end); coordinates are of the top left corner of the path-corner; y_start must be less than y_end

    # Precomputed Constants
    PATH_EDGES: chex.Array = jnp.concatenate((HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES), axis=0)
    RECTANGLES: chex.Array = jnp.array(calculate_rectangles(HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES, PATH_EDGES, PATH_CORNERS))
    RECTANGLE_CORNERS: chex.Array = jax.vmap(precompute_rectangle_corners, in_axes=(0, None))(RECTANGLES, PATH_EDGES)
    PATH_MASK, RENDERING_PATH_MASK = generate_path_mask(WIDTH, HEIGHT, HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES, jnp.full((HORIZONTAL_PATH_EDGES.shape[0],), True), jnp.full((VERTICAL_PATH_EDGES.shape[0],), True))  # Path mask are the single lines which restrict the movement, while rendering path mask includes the width of the paths for rendering
    PATH_PATTERN, WALKED_ON_PATTERN = generate_path_pattern(WIDTH, HEIGHT, PATH_COLOR, WALKED_ON_COLOR)
    PATH_SPRITE: chex.Array = jnp.where(RENDERING_PATH_MASK[:, :, None] == 1, PATH_PATTERN, jnp.full((HEIGHT, WIDTH, 4), 0, dtype=jnp.uint8))

# immutable state container
class AmidarState(NamedTuple):
    random_key: chex.Array  # Random key for JAX operations
    freeze_counter: chex.Array  # Counter for freezing the game
    score: chex.Array
    lives: chex.Array
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array # 0=up, 1=left, 2=down, 3=right
    last_walked_corner: chex.Array # (2,) -> (x, y) of the last corner walked on
    walked_on_paths: chex.Array
    completed_rectangles: chex.Array
    enemy_positions: chex.Array # (MAX_ENEMIES, 2) -> (x, y) for each enemy
    enemy_directions: chex.Array # (MAX_ENEMIES, 1) -> direction of each enemy (0=up, 1=left, 2=down, 3=right)
    enemy_types: chex.Array # (MAX_ENEMIES, 1) -> type of each enemy (Warrior, Pig, Shadow, Chicken)

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class AmidarObservation(NamedTuple):
    player: EntityPosition
    warriors: chex.Array
    # pigs: chex.Array
    # shadows: chex.Array
    # chickens: chex.Array

class AmidarInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

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
        new_x = state.player_x + jnp.where(direction == 1, -1, 0) + jnp.where(direction == 3, 1, 0)
        new_y = state.player_y + jnp.where(direction == 0, -1, 0) + jnp.where(direction == 2, 1, 0)
        # only move if new position is on the path
        new_x = jnp.where(on_path(new_x, new_y), new_x, state.player_x)
        new_y = jnp.where(on_path(new_x, new_y), new_y, state.player_y)
        return new_x, new_y
    
    has_not_moved = jnp.logical_and(new_x == state.player_x, new_y == state.player_y)
    movement_key_pressed = jnp.logical_or(up, jnp.logical_or(down, jnp.logical_or(left, right)))
    new_x, new_y = jax.lax.cond(jnp.logical_and(has_not_moved, movement_key_pressed), move_in_previous_direction, lambda direction: (new_x, new_y), state.player_direction)
        
    player_direction = jnp.select([new_y < state.player_y, new_x < state.player_x, new_y > state.player_y, new_x > state.player_x],
                                  [0, 1, 2, 3], default=state.player_direction)
    
    # Check if the new position is a corner, in which case check if a new path edge is walked on

    def corner_handeling():
        # Set the walked on paths
        # last walked corner -> (new_x, new_y)
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
        match_index = find_edge(jnp.array([[new_x, new_y], state.last_walked_corner]), constants.PATH_EDGES)

        def score_points():
            """Scores points based on the edge walked on."""
            # Calculate the points scored based on the edge walked on
            edge = constants.PATH_EDGES[match_index]
            points_scored = jax.lax.cond(edge[0, 0] == edge[1, 0],  # Vertical edge
                         lambda: ((edge[1, 1] - edge[0, 1]) // constants.PIXELS_PER_POINT_VERTICAL),  # Vertical edge
                         lambda: ((edge[1, 0] - edge[0, 0]) // constants.PIXELS_PER_POINT_HORIZONTAL))  # Horizontal edge
            # Ensure points_scored is an integer
            points_scored = points_scored.astype(jnp.int32)
            # Update the walked on paths
            walked_on_paths = state.walked_on_paths.at[match_index].set(1)

            def check_rectangle_completion():
                """Checks if a rectangle is completed and updates the completed rectangles."""
                completed_rectangles = state.completed_rectangles
                
                # Create a mask for rectangles that contain this edge
                rectangles_containing_edge_mask = constants.RECTANGLES[:, match_index] == 1

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
                                                 jnp.logical_not(state.completed_rectangles))
                
                new_rectangle_completed = jnp.any(new_completions)
                
                # Update completed rectangles
                completed_rectangles = jnp.logical_or(state.completed_rectangles, new_completions)
                
                return new_rectangle_completed, completed_rectangles
            
            # call check_rectangle_completion to check if a rectangle is completed
            new_rectangle_completed, completed_rectangles = check_rectangle_completion()
            # Add bonus points for completing a rectangle
            points_scored = jax.lax.cond(new_rectangle_completed, lambda: points_scored + constants.BONUS_POINTS_PER_RECTANGLE, lambda: points_scored)

            return points_scored, walked_on_paths, completed_rectangles

        # If the edge is not walked on yet, score points and update the walked on paths & completed rectangles
        points_scored, walked_on_paths, completed_rectangles = jax.lax.cond(jnp.logical_and(match_index >= 0, state.walked_on_paths[match_index] == 0), score_points, lambda: (0, state.walked_on_paths, state.completed_rectangles))

        last_walked_corner = jnp.array([new_x, new_y])   # Update the last walked corner to the new position
        return points_scored, last_walked_corner, walked_on_paths, completed_rectangles


    is_corner = jnp.any(jnp.all(constants.PATH_CORNERS == jnp.array([new_x, new_y]), axis=1))
    points_scored, last_walked_corner, walked_on_paths, completed_rectangles = jax.lax.cond(is_corner, corner_handeling, lambda: (0, state.last_walked_corner, state.walked_on_paths, state.completed_rectangles))

    # TODO: Add checking if all edges(next level)/corner edges(chickens) are walked on
    return points_scored, new_x, new_y, player_direction, last_walked_corner, walked_on_paths, completed_rectangles

def enemies_step(constants: AmidarConstants, state: AmidarState, random_key: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Updates the enemy positions based on their behavior."""

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

    def calculate_good_enemy_directions(enemy_direction, possible_directions):
        """From the possible movement directions exclude u-turns"""
        opposite_direction = (enemy_direction + 2) % 4
        good_directions = possible_directions.at[opposite_direction].set(False)  # Exclude the opposite direction (u-turn)
        return good_directions

    good_directions_per_enemy = jax.vmap(calculate_good_enemy_directions, in_axes=0)(state.enemy_directions, possible_directions_per_enemy)

    def move_enemy(enemy_x, enemy_y, direction, possible_directions, good_directions, random_key):
        """Chooses randomly from the possible directions, only taking u-turns when necassary."""

        def choose_direction():
            # If there are good directions, choose one, else choose from all possible directions
            chosen_direction = jax.lax.cond(jnp.any(good_directions),
                                            lambda: jax.random.choice(random_key, jnp.arange(4), shape=(), p=jnp.where(good_directions, 1/jnp.count_nonzero(good_directions), 0)),
                                            lambda: jax.random.choice(random_key, jnp.arange(4), shape=(), p=jnp.where(possible_directions, 1/jnp.count_nonzero(possible_directions), 0)))
            return chosen_direction

        # if no directions are possible stay with the current one
        chosen_direction = jax.lax.cond(jnp.any(possible_directions), choose_direction, lambda: direction)
        
        new_x, new_y = jax.lax.cond(jnp.any(possible_directions),
            lambda: (enemy_x + jnp.where(chosen_direction == 1, -1, 0) + jnp.where(chosen_direction == 3, 1, 0),
                     enemy_y + jnp.where(chosen_direction == 0, -1, 0) + jnp.where(chosen_direction == 2, 1, 0)),
            lambda: (enemy_x, enemy_y))  # If the direction is not possible, stay in place
        return jnp.array([new_x, new_y]), chosen_direction

    new_enemy_positions, new_enemy_directions = jax.vmap(move_enemy, in_axes=0)(state.enemy_positions[:, 0], state.enemy_positions[:, 1], state.enemy_directions, possible_directions_per_enemy, good_directions_per_enemy, enemy_keys)

    return new_enemy_positions, new_enemy_directions

def check_for_collisions(constants, player_x, player_y, enemy_positions, enemy_types) -> bool:
    """Checks if the player collides with any enemy."""
    # the collision is counted if any part of the sprites touch, so per enemy we need to check if the sprite overlaps the players

    def check_enemy_for_collision(enemy_x, enemy_y, enemy_type):
        """Checks if the enemy collides with the player."""
        collision = jax.lax.cond(
            enemy_type == 0, # shadows don't collide with the player
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

def handle_collisions(enemy_types: chex.Array, collisions: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Handles the collisions with the enemies.
    Returns the new enemy positions, directions, types, lives and points scored."""

    def handle_collision(enemy_type, collision):
        """Handles the collision with a single enemy."""
        enemy_type, points, lost_live = jax.lax.cond(collision, lambda: (enemy_type, 0, 1), lambda: (enemy_type, 0, 0))
        return enemy_type, points, lost_live

    enemy_types, points_scored, lost_live = jax.vmap(handle_collision, in_axes=(0, 0))(enemy_types, collisions)

    lost_live = jnp.any(lost_live)  # If any enemy collision results in a lost life, set lost_live to True
    points_scored = jnp.sum(points_scored)  # Sum the points scored from all collisions

    return enemy_types, points_scored, lost_live

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

    def reset(self, key=None) -> Tuple[AmidarObservation, AmidarState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = AmidarState(
            random_key=jax.random.key(self.constants.RANDOM_KEY_SEED),
            freeze_counter=jnp.array(self.constants.FREEZE_DURATION).astype(jnp.int32),  # Freeze counter for the initial freeze
            score=jnp.array(0).astype(jnp.int32),  # Initial score
            lives=jnp.array(self.constants.INITIAL_LIVES).astype(jnp.int32),  # Initial lives
            player_x=jnp.array(self.constants.INITIAL_PLAYER_POSITION[0]).astype(jnp.int32),
            player_y=jnp.array(self.constants.INITIAL_PLAYER_POSITION[1]).astype(jnp.int32),
            player_direction=jnp.array(self.constants.INITIAL_PLAYER_DIRECTION).astype(jnp.int32),
            last_walked_corner=jnp.array([0, 0]).astype(jnp.int32),  # Last corner walked on
            walked_on_paths=(jnp.zeros(jnp.shape(self.constants.PATH_EDGES)[0], dtype=jnp.int32)).at[self.constants.PLAYER_STARTING_PATH].set(1),  # Initialize walked on paths
            completed_rectangles=jnp.zeros(jnp.shape(self.constants.RECTANGLES)[0], dtype=jnp.bool_),  # Initialize completed rectangles
            enemy_positions=self.constants.INITIAL_ENEMY_POSITIONS,
            enemy_directions=self.constants.INITIAL_ENEMY_DIRECTIONS,
            enemy_types=self.constants.INITIAL_ENEMY_TYPES,
        )
        initial_obs = self._get_observation(state)

        # jnp.set_printoptions(threshold=jnp.inf)
        # jax.debug.print("{m}", m=PATH_MASK)

        return initial_obs, state
    
    # TODO check these notes and see what they do/should do 
    # NOTE: added this for compatibility (its incorrect, I was just too lazy to get the correct one!)
    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.constants.HEIGHT, self.constants.WIDTH, 4), dtype=jnp.uint8)

    # NOTE: added this for compatibility
    def state_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.constants.HEIGHT, self.constants.WIDTH, 4), dtype=jnp.uint8)

    # NOTE: added this for compatibility
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    # NOTE: added this for compatibility (see changes from a few weeks ago), it just calls the renderer as a class attribute since that was more comfortable when writing wrappers and such (i.e. easier for me xD)
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AmidarState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AmidarState, action: chex.Array) -> Tuple[AmidarObservation, AmidarState, float, bool, AmidarInfo]:
            
        def take_step():
            random_key, random_enemies_key = jax.random.split(state.random_key)

            player_state = player_step(self.constants, state, action)
            (points_scored_player_step, player_x, player_y, player_direction, last_walked_corner, walked_on_paths, completed_rectangles) = player_state

            enemies_state = enemies_step(self.constants, state, random_enemies_key)
            (enemy_positions, enemy_directions) = enemies_state

            enemy_types = state.enemy_types  # For now, keep the enemy types the same, TODO later add jump and chicken activation/deactivation
            # TODO add next level handling

            # Check for collisions with enemies
            collisions = check_for_collisions(self.constants, player_x, player_y, enemy_positions, enemy_types)
            # jax.debug.print("Collisions: {collisions}", collisions=collisions)
            (enemy_types, points_scored_enemy_collision, lost_live) = jax.lax.cond(
                jnp.any(collisions),
                lambda: handle_collisions(enemy_types, collisions),
                lambda: (enemy_types, 0, False)
            )

            # Reset positions if a life is lost 
            enemy_positions, enemy_directions, player_x, player_y, player_direction, lives, freeze_counter = jax.lax.cond(lost_live,
                lambda: (self.constants.INITIAL_ENEMY_POSITIONS, self.constants.INITIAL_ENEMY_DIRECTIONS, self.constants.INITIAL_PLAYER_POSITION[0], self.constants.INITIAL_PLAYER_POSITION[1], self.constants.INITIAL_PLAYER_DIRECTION, state.lives-1, self.constants.FREEZE_DURATION),  # Reset enemy & player positions and directions, decrement lives
                lambda: (enemy_positions, enemy_directions, player_x, player_y, player_direction, state.lives, state.freeze_counter))  # Keep the current enemy positions and directions

            # TODO where do i change the enemy type? (jump, jump time up, collision with chicken, chicken time up, chicken activated, next level)

            # Check for game over condition
            # If lives are less than 0, the game is over
            done, lives = jax.lax.cond(
                lives < 0,
                lambda: (True, 0),
                lambda: (False, lives)
            )

            new_state = AmidarState(
                random_key=random_key,
                freeze_counter=freeze_counter,
                score=state.score + points_scored_player_step + points_scored_enemy_collision, # Could change in multiple functions, so it is not calculated in the function based on the previous state
                lives=lives,
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                last_walked_corner=last_walked_corner,
                walked_on_paths=walked_on_paths,
                completed_rectangles=completed_rectangles,
                enemy_positions=enemy_positions,
                enemy_directions=enemy_directions,
                enemy_types=enemy_types,
            )
            env_reward = 0.0
            info = AmidarInfo(
                time=jnp.array(0),
                all_rewards=jnp.array(0.0),
            )
            
            observation = self._get_observation(new_state)
            return observation, new_state, env_reward, done, info
        
        # If the game is frozen, just return the current state
        def freeze_game():
            # Decrement the freeze counter
            new_freeze_counter = state.freeze_counter - 1
            new_state = state._replace(freeze_counter=new_freeze_counter)
            return self._get_observation(new_state), new_state, 0.0, False, AmidarInfo(time=jnp.array(0), all_rewards=jnp.array(0.0))
        
        # Check if the game is frozen
        is_frozen = state.freeze_counter > 0
        # jax.debug.print("Freeze counter: {freeze_counter}", freeze_counter=state.freeze_counter)
        observation, new_state, env_reward, done, info = jax.lax.cond(is_frozen, freeze_game, take_step)
        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AmidarState):
        # create player
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.constants.PLAYER_SIZE[0]),
            height=jnp.array(self.constants.PLAYER_SIZE[1]),
        )

        # Define a function to convert enemy positions to entity format
        def convert_to_entity(pos, size):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                #pos[2] != 0,  # active flag TODO remove?
            ])

        # Apply conversion to each type of entity using vmap

        # warriors
        warriors = jax.vmap(lambda pos: convert_to_entity(pos, self.constants.ENEMY_SIZE))(state.enemy_positions)

        return AmidarObservation(
            player=player,
            warriors=warriors
        )
        
    
    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

def load_sprites():
    """Load all sprites required for Amidar rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites

    DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/amidar/score/{}.npy"))

    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/player_ghost.npy"))

    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/background.npy"), transpose=True)

    life = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/life.npy"))

    warrior = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/enemy/warrior.npy"))

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_LIFE = jnp.expand_dims(life, axis=0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_WARRIOR = jnp.expand_dims(warrior, axis=0)

    return (
        SPRITE_BG,
        DIGITS,
        SPRITE_LIFE,
        SPRITE_PLAYER,
        SPRITE_WARRIOR,
    )


class AmidarRenderer(JAXGameRenderer):
    """JAX-based Amidar game renderer, optimized with JIT compilation."""

    def __init__(self, constants: AmidarConstants = None):
        self.constants = constants or AmidarConstants()
        (
            self.SPRITE_BG,
            self.DIGITS,
            self.SPRITE_LIFE,
            self.SPRITE_PLAYER,
            self.SPRITE_WARRIOR,
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

        # Render background - (0, 0) is top-left corner
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render paths
        raster = aj.render_at(raster, 0, 0, self.constants.PATH_SPRITE)

        # Render walked on paths
        walked_on_paths_horizontal = state.walked_on_paths[0:jnp.shape(self.constants.HORIZONTAL_PATH_EDGES)[0]]
        walked_on_paths_vertical = state.walked_on_paths[jnp.shape(self.constants.HORIZONTAL_PATH_EDGES)[0]:]
        _, walked_on_rendering_mask = generate_path_mask(self.constants.WIDTH, self.constants.HEIGHT, self.constants.HORIZONTAL_PATH_EDGES, self.constants.VERTICAL_PATH_EDGES, horizontal_cond=walked_on_paths_horizontal, vertical_cond=walked_on_paths_vertical)
        walked_on_paths_sprite = jnp.where(walked_on_rendering_mask[:, :, None] == 1, self.constants.WALKED_ON_PATTERN, jnp.full((self.constants.HEIGHT, self.constants.WIDTH, 4), 0, dtype=jnp.uint8))
        raster = aj.render_at(raster, 0, 0, walked_on_paths_sprite)

        # Render completed rectangles
        # Still computed in the (HEIGHT, WIDTH) format and transposed later since the code uses (x, y) coordinates
        # Create coordinate grids for vectorized operations
        x_coords, y_coords = jnp.meshgrid(jnp.arange(self.constants.WIDTH), jnp.arange(self.constants.HEIGHT), indexing='ij')
        completed_rectangles_mask = jnp.zeros((self.constants.WIDTH, self.constants.HEIGHT), dtype=jnp.uint8)
        def create_rectangle_mask(i, mask):
            rect_min_x, rect_min_y, rect_max_x, rect_max_y = self.constants.RECTANGLE_CORNERS[i]
            is_completed = state.completed_rectangles[i]
            def create_mask():
                in_rect_x = jnp.logical_and(x_coords >= rect_min_x, x_coords <= rect_max_x)
                in_rect_y = jnp.logical_and(y_coords >= rect_min_y, y_coords <= rect_max_y)
                rect_mask = jnp.logical_and(in_rect_x, in_rect_y).astype(jnp.uint8)
                return jnp.logical_or(mask, rect_mask).astype(jnp.uint8)
            return jax.lax.cond(is_completed, create_mask, lambda: mask)
        completed_rectangles_mask = jax.lax.fori_loop(0, jnp.shape(self.constants.RECTANGLES)[0], create_rectangle_mask, completed_rectangles_mask)
        completed_rectangles_mask = jnp.transpose(completed_rectangles_mask, (1, 0))  # Transpose to match the HWC format for rendering

        completed_rectangles_sprite = jnp.where(completed_rectangles_mask[:, :, None] == 1, self.constants.WALKED_ON_PATTERN, jnp.full((self.constants.HEIGHT, self.constants.WIDTH, 4), 0, dtype=jnp.uint8))
        raster = aj.render_at(raster, 0, 0, completed_rectangles_sprite)

        # Render score
        score_array = aj.int_to_digits(state.score, max_digits=8)
        # convert the score to a list of digits
        number_of_digits = (jnp.log10(state.score)+1).astype(jnp.int32)
        number_of_digits = jnp.maximum(number_of_digits, 1)  # Ensure at least one digit is rendered
        def render_char(i, current_raster):
            # i is the loop index (0 up to num_to_render-1)
            digit_index_in_array = 8 - number_of_digits + i
            digit_value = score_array[digit_index_in_array]
            sprite_to_render = self.DIGITS[digit_value] # Gets (W, H, C) sprite
            render_x = 103-(number_of_digits * 8) + i * 8 # Calculate x position based on loop index
            return aj.render_at(current_raster, render_x, 176, sprite_to_render)

        raster = jax.lax.fori_loop(0, number_of_digits, render_char, raster)

        # Render lives
        lives = state.lives
        def render_lives(i, carry):
            current_raster, x, y = carry
            # Render a life sprite at the specified position
            frame_life = aj.get_sprite_frame(self.SPRITE_LIFE, 0)
            return aj.render_at(current_raster, x, y, frame_life), x-16, y  # Move x left by 16 pixels for the next life
        raster, _, _ = jax.lax.fori_loop(0, lives, render_lives, (raster, 148, 175))

        # Render player - IMPORTANT: Swap x and y coordinates
        # render_at takes (raster, y, x, sprite) but we need to swap them due to transposition
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = aj.render_at(raster, state.player_x+self.constants.PLAYER_SPRITE_OFFSET[0], state.player_y+self.constants.PLAYER_SPRITE_OFFSET[1], frame_player)

        # Render enemies - IMPORTANT: Swap x and y coordinates
        # TODO differentiate enemy types and if they should be rendered or not
        frame_enemy = aj.get_sprite_frame(self.SPRITE_WARRIOR, 0)
        
        # Use scan to accumulate the raster updates
        def scan_render_enemy(raster, enemy_pos):
            new_raster = aj.render_at(raster, enemy_pos[0]+self.constants.ENEMY_SPRITE_OFFSET[0], enemy_pos[1]+self.constants.ENEMY_SPRITE_OFFSET[1], frame_enemy)
            return new_raster, None
        
        raster, _ = jax.lax.scan(scan_render_enemy, raster, state.enemy_positions)

        ###### For DEBUGGING #######

        # # Render path edges
        # def render_path(i, raster):
        #     path = jnp.array(self.constants.PATH_EDGES[i])
        #     raster = render_line(raster, path, (255, 0, 0))
        #     return raster
        # raster = jax.lax.fori_loop(0, jnp.shape(self.constants.PATH_EDGES)[0], render_path, raster)

        # # Render path mask
        # all_white = jnp.full_like(raster, 255, dtype=jnp.uint8)
        # raster = jnp.where(self.constants.PATH_MASK[:, :, None] == 1, all_white, raster)

        # # Render rendering path mask
        # all_white = jnp.full_like(raster, 255, dtype=jnp.uint8)
        # raster = jnp.where(self.constants.RENDERING_PATH_MASK[:, :, None] == 1, all_white, raster)

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

        return raster

@jax.jit
def render_line(raster, coords, color):
    """Renders a line on the raster from coords [[x1, y1], [x2, y2]] with the given color.

    Args:
        raster: JAX array of shape (Width, Height, Channels) for the target image.
        coords: JAX array of shape (2, 2), where coords[0] = [x1, y1] and coords[1] = [x2, y2].
        color: RGB or RGBA tuple/list/array for the line color.

    Returns:
        Updated raster with the line rendered.
    """
    color = jnp.asarray(color, dtype=jnp.uint8)  # Ensure color is a JAX array
    if color.shape[0] not in (3, 4):
        raise ValueError("Color must be an RGB or RGBA array.")

    coords = jnp.asarray(coords, dtype=jnp.int32)  # Ensure coords are JAX arrays
    x1, y1 = coords[0]
    x2, y2 = coords[1]

    # Compute Bresenham's algorithm parameters
    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)
    sx = jnp.where(x1 < x2, 1, -1)
    sy = jnp.where(y1 < y2, 1, -1)
    err = dx - dy

    # Total number of steps
    num_steps = dx + dy + 1

    def loop_body(i, carry):
        raster, x, y, err = carry

        # Clip coordinates to raster bounds
        raster_width, raster_height, _ = raster.shape
        raster = jax.lax.cond(jnp.logical_and(jnp.logical_and(0 <= x, x < raster_width), jnp.logical_and(0 <= y, y < raster_height)), 
                     lambda raster: raster.at[x, y, :color.shape[0]].set(color), lambda raster: raster, raster)

        # Update Bresenham's algorithm variables
        e2 = 2 * err
        new_err = jnp.where(e2 > -dy, err - dy, err)
        new_x = jnp.where(e2 > -dy, x + sx, x)
        new_err = jnp.where(e2 < dx, new_err + dx, new_err)
        new_y = jnp.where(e2 < dx, y + sy, y)

        return raster, new_x, new_y, new_err

    # Use fori_loop to iterate
    raster, _, _, _ = jax.lax.fori_loop(
        0, num_steps, loop_body, (raster, x1, y1, err)
    )

    return raster

if __name__ == "__main__":
    print("please use the play script to run the game")