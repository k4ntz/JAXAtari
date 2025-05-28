

from functools import partial
import os
from typing import NamedTuple, Tuple
import chex
import jax
import jax.numpy as jnp
import pygame
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj

WIDTH = 160
HEIGHT = 210

# Object sizes (width, height)
PLAYER_SIZE = (7, 7)
ENEMY_SIZE = (7, 7)

MAX_ENEMIES = 6

INITIAL_ENEMY_POSITIONS = jnp.array(
    [
        [15, 14],  # Enemy 1
        [15, 14],  # Enemy 2
        [43, 14],  # Enemy 3
        [15, 137],  # Enemy 4
        [15, 160],  # Enemy 5
        [51, 160],  # Enemy 6
    ]
)
INITIAL_ENEMY_TYPES = jnp.array([1, 1, 1, 1, 1, 1])

PATH_CORNERS = jnp.array( #TODO remove
    [
        [16, 15],
        [40, 15],
        [56, 15],
        [72, 15],
        [84, 15],
        [100, 15],
        [116, 15],
        [140, 15],
        [16, 45],
        [40, 45],
        [56, 45],
        [72, 45],
        [84, 45],
        [100, 45],
        [116, 45],
        [140, 45],
        [32, 45],
        [52, 45],
        [64, 45],
        [92, 45],
        [104, 45],
        [124, 45],
        [16, 75],
        [32, 75],
        [52, 75],
        [64, 75],
        [92, 75],
        [104, 75],
        [124, 75],
        [140, 75],
        [28, 75],
        [60, 75],
        [96, 75],
        [128, 75],
        [16, 105],
        [28, 105],
        [60, 105],
        [96, 105],
        [128, 105],
        [140, 105],
        [36, 105],
        [72, 105],
        [84, 105],
        [120, 105],
        [16, 135],
        [36, 135],
        [72, 135],
        [84, 135],
        [120, 135],
        [140, 135],
        [40, 135],
        [64, 135],
        [92, 135],
        [116, 135],
        [16, 165],
        [40, 165],
        [64, 165],
        [92, 165],
        [116, 165],
        [140, 165]
    ]
)

HORIZONTAL_PATH_EDGES = jnp.array(
    [
         # Tuples of (start, end), coordinates are of the top left corner of the path-corner
        [[16, 15], [40, 15]],
        [[40, 15], [56, 15]],
        [[56, 15], [72, 15]],
        [[72, 15], [84, 15]],
        [[84, 15], [100, 15]],
        [[100, 15], [116, 15]],
        [[116, 15], [140, 15]],
        [[16, 45], [32, 45]],
        [[32, 45], [40, 45]],
        [[40, 45], [52, 45]],
        [[52, 45], [56, 45]],
        [[56, 45], [64, 45]],
        [[64, 45], [72, 45]],
        [[72, 45], [84, 45]],
        [[84, 45], [92, 45]],
        [[92, 45], [100, 45]],
        [[100, 45], [104, 45]],
        [[104, 45], [116, 45]],
        [[116, 45], [124, 45]],
        [[124, 45], [140, 45]],
        [[16, 75], [28, 75]],
        [[28, 75], [32, 75]],
        [[32, 75], [52, 75]],
        [[52, 75], [60, 75]],
        [[60, 75], [64, 75]],
        [[64, 75], [92, 75]],
        [[92, 75], [96, 75]],
        [[96, 75], [104, 75]],
        [[104, 75], [124, 75]],
        [[124, 75], [128, 75]],
        [[128, 75], [140, 75]],
        [[16, 105], [28, 105]],
        [[28, 105], [36, 105]],
        [[36, 105], [60, 105]],
        [[60, 105], [72, 105]],
        [[72, 105], [84, 105]],
        [[84, 105],  [96, 105]],
        [[96, 105], [120, 105]],
        [[120, 105], [128, 105]],
        [[128, 105], [140, 105]],
        [[16, 135], [36, 135]],
        [[36, 135], [40, 135]],
        [[40, 135], [64, 135]],
        [[64, 135], [72, 135]],
        [[72, 135], [84, 135]],
        [[84, 135], [92, 135]],
        [[92, 135], [116, 135]],
        [[116, 135], [120, 135]],
        [[120, 135], [140, 135]],
        [[16, 165], [40, 165]],
        [[40, 165], [64, 165]],
        [[64, 165], [92, 165]],
        [[92, 165], [116, 165]],
        [[116, 165], [140, 165]],
    ]
)

VERTICAL_PATH_EDGES = jnp.array(
    [
        # Tuples of (start, end), coordinates are of the top left corner of the path-corner
        [[16, 15], [16, 45]],
        [[16, 45], [16, 75]],
        [[16, 75], [16, 105]],
        [[16, 105], [16, 135]],
        [[16, 135], [16, 165]],
        [[28, 75], [28, 105]],
        [[32, 45], [32, 75]],
        [[36, 105], [36, 135]],
        [[40, 15], [40, 45]],
        [[40, 135], [40, 165]],
        [[52, 45], [52, 75]],
        [[56, 15], [56, 45]],
        [[60, 75], [60, 105]],
        [[64, 45], [64, 75]],
        [[64, 135], [64, 165]],
        [[72, 15], [72, 45]],
        [[72, 105], [72, 135]],
        [[84, 15], [84, 45]],
        [[84, 105], [84, 135]],
        [[92, 45], [92, 75]],
        [[92, 135], [92, 165]],
        [[96, 75], [96, 105]],
        [[100, 15], [100, 45]],
        [[104, 45], [104, 75]],
        [[116, 15], [116, 45]],
        [[116, 135], [116, 165]],
        [[120, 105], [120, 135]],
        [[124, 45], [124, 75]],
        [[128, 75], [128, 105]],
        [[140, 15], [140, 45]],
        [[140, 45], [140, 75]],
        [[140, 75], [140, 105]],
        [[140, 105], [140, 135]],
        [[140, 135], [140, 165]]
    ]
)

PATH_EDGES = jnp.concatenate((HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES), axis=0)

@jax.jit
def generate_path_mask():
    """Generates a mask for the path edges using Bresenham's line algorithm.
    Args:
        path_edges: JAX array of shape (N, 2, 2) representing the path edges.
    """
    # Create an empty mask 
    mask = jnp.zeros((WIDTH, HEIGHT), dtype=jnp.int32)

    def add_horizontal_edge(i, mask):
        start, end = HORIZONTAL_PATH_EDGES[i]

        x1, y1 = start
        x2, y2 = end

        def loop(x, carry):
            mask, y = carry
            mask = mask.at[x, y].set(1)
            return (mask, y)
        
        mask, _ = jax.lax.fori_loop(x1, x2 + 1, loop, (mask, y1))
        return mask
    
    def add_vertical_edge(i, mask):
        start, end = VERTICAL_PATH_EDGES[i]

        x1, y1 = start
        x2, y2 = end

        def loop(y, carry):
            mask, x = carry
            mask = mask.at[x, y].set(1)
            return (mask, x)
        
        mask, _ = jax.lax.fori_loop(y1, y2 + 1, loop, (mask, x1))
        return mask
    
    mask = jax.lax.fori_loop(0, jnp.shape(HORIZONTAL_PATH_EDGES)[0], add_horizontal_edge, mask)
    mask = jax.lax.fori_loop(0, jnp.shape(VERTICAL_PATH_EDGES)[0], add_vertical_edge, mask)

    return mask

PATH_MASK = generate_path_mask()

# immutable state container
class AmidarState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array # 0=up, 1=down, 2=left, 3=right
    enemy_positions: chex.Array # (MAX_ENEMIES, 2) -> (x, y) for each enemy TODO possibly add direction?
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

def player_step(state: AmidarState, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Updates the player position based on the action taken.
    Returns the new player x and y coordinates and the direction."""

    def on_path(x, y):
        """Checks if the given coordinates are on the path."""
        # add 1 to x and y to account for the offset of the top left corner of the player sprite in relation to the path mask
        return PATH_MASK[x+1, y+1] == 1

    up = jnp.logical_or( action == Action.UP, action == Action.UPFIRE)
    down = jnp.logical_or( action == Action.DOWN, action == Action.DOWNFIRE)
    left = jnp.logical_or( action == Action.LEFT, action == Action.LEFTFIRE)
    right = jnp.logical_or( action == Action.RIGHT, action == Action.RIGHTFIRE)

    new_x = state.player_x + jnp.where(left, -1, 0) + jnp.where(right, 1, 0)
    new_y = state.player_y + jnp.where(up, -1, 0) + jnp.where(down, 1, 0)

    # jax.debug.print("new player position: ({}, {})", new_x, new_y)
    # jax.debug.print("Path mask at new position: {}", PATH_MASK[new_x, new_y])

    new_x = jnp.where(on_path(new_x, new_y), new_x, state.player_x)
    new_y = jnp.where(on_path(new_x, new_y), new_y, state.player_y)

    # if the direction is not possiple to move in, try moving in the previous direction
    def move_in_previous_direction(direction):
        new_x = state.player_x + jnp.where(direction == 2, -1, 0) + jnp.where(direction == 3, 1, 0)
        new_y = state.player_y + jnp.where(direction == 0, -1, 0) + jnp.where(direction == 1, 1, 0)
        # only move if new position is on the path
        new_x = jnp.where(on_path(new_x, new_y), new_x, state.player_x)
        new_y = jnp.where(on_path(new_x, new_y), new_y, state.player_y)
        return new_x, new_y
    
    has_not_moved = jnp.logical_and(new_x == state.player_x, new_y == state.player_y)
    movement_key_pressed = jnp.logical_or(up, jnp.logical_or(down, jnp.logical_or(left, right)))
    new_x, new_y = jax.lax.cond(jnp.logical_and(has_not_moved, movement_key_pressed), move_in_previous_direction, lambda direction: (new_x, new_y), state.player_direction)
        
    player_direction = jnp.select([new_x > state.player_x, new_x < state.player_x, new_y > state.player_y, new_y < state.player_y],
                                  [3, 2, 1, 0], default=state.player_direction)

    # TODO: correct allignment of the player on the path on the bottom (maybe move the path if this aplies to the enemies equally)
    # TODO: replicate slight stopping at corners 
    # TODO: check if the behavior is correct/according to the original game
    return new_x, new_y, player_direction


class JaxAmidar(JaxEnvironment[AmidarState, AmidarObservation, AmidarInfo]):
    def __init__(self, reward_funcs: list[callable]=None):
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
        self.obs_size = 4 #TODO add as needed

    def reset(self, key=None) -> Tuple[AmidarObservation, AmidarState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = AmidarState(
            player_x=jnp.array(139).astype(jnp.int32),
            player_y=jnp.array(88).astype(jnp.int32),
            player_direction=jnp.array(0).astype(jnp.int32),
            enemy_positions=INITIAL_ENEMY_POSITIONS,
            enemy_types=INITIAL_ENEMY_TYPES,
        )
        initial_obs = self._get_observation(state)

        # jnp.set_printoptions(threshold=jnp.inf)
        # jax.debug.print("{m}", m=PATH_MASK)

        return initial_obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AmidarState, action: chex.Array) -> Tuple[AmidarObservation, AmidarState, float, bool, AmidarInfo]:
        observation = self._get_observation(state)
        player_state = player_step(state, action)
        (player_x, player_y, player_direction) = player_state
        new_state = AmidarState(
            player_x=player_x,
            player_y=player_y,
            player_direction=player_direction,
            enemy_positions=state.enemy_positions,
            enemy_types=state.enemy_types,
        )
        env_reward = 0.0
        done = False
        info = AmidarInfo(
            time=jnp.array(0),
            all_rewards=jnp.array(0.0),
        )
        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AmidarState):
        # create player
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
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
        warriors = jax.vmap(lambda pos: convert_to_entity(pos, ENEMY_SIZE))(state.enemy_positions)

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
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/player_ghost.npy"), transpose=True)

    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/background.npy"), transpose=True)

    paths = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/paths.npy"), transpose=True)

    warrior = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/amidar/enemy/warrior.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_PATHS = jnp.expand_dims(paths, axis=0)
    SPRITE_WARRIOR = jnp.expand_dims(warrior, axis=0)

    return (
        SPRITE_BG,
        SPRITE_PATHS,
        SPRITE_PLAYER,
        SPRITE_WARRIOR,
    )


class AmidarRenderer(AtraJaxisRenderer):
    """JAX-based Amidar game renderer, optimized with JIT compilation."""

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PATHS,
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
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Render background - (0, 0) is top-left corner
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render paths - The Top Left corner of the path is (16, 15)
        # TODO render the paths that have been walked on
        # TODO make adaptable to different configurations?
        frame_paths = aj.get_sprite_frame(self.SPRITE_PATHS, 0)
        raster = aj.render_at(raster, 16, 15, frame_paths)

        # Render player - IMPORTANT: Swap x and y coordinates
        # render_at takes (raster, y, x, sprite) but we need to swap them due to transposition
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = aj.render_at(raster, state.player_x, state.player_y, frame_player)

        # Render enemies - IMPORTANT: Swap x and y coordinates
        # TODO differevtiatte enemy types and if they should be rendered or not
        frame_enemy = aj.get_sprite_frame(self.SPRITE_WARRIOR, 0)
        enemy_positions = state.enemy_positions

        def render_enemy(i, raster):
            return aj.render_at(raster, enemy_positions[i][0], enemy_positions[i][1], frame_enemy)
        
        raster = jax.lax.fori_loop(0, MAX_ENEMIES, render_enemy, raster)

        ###### For DEBUGGING #######

        # Render path edges
        def render_path(i, raster):
            path = jnp.array(PATH_EDGES[i])

            raster = render_line(raster, path, (255, 0, 0))

            return raster

        raster = jax.lax.fori_loop(0, jnp.shape(PATH_EDGES)[0], render_path, raster)

        #render mask
        all_white = jnp.full_like(raster, 255, dtype=jnp.uint8)
        raster = jnp.where(PATH_MASK[:, :, None] == 1, all_white, raster)

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