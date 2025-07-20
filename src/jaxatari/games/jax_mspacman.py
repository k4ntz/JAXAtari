# Group: Sooraj Rathore, Kadir Özen

from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.games.mspacman_mazes import MAZES, load_background, pacman_rgba, load_ghosts

from jax import random, Array



WIDTH = 160
HEIGHT = 210

BASE_LEVEL = 0


def last_pressed_action(action, prev_action):
    """
    Returns the last pressed action in cases where both actions are pressed
    """
    if action == Action.UPRIGHT:
        if prev_action == Action.UP:
            return Action.RIGHT
        else:
            return Action.UP
    elif action == Action.UPLEFT:
        if prev_action == Action.UP:
            return Action.LEFT
        else:
            return Action.UP
    elif action == Action.DOWNRIGHT:
        if prev_action == Action.DOWN:
            return Action.RIGHT
        else:
            return Action.DOWN
    elif action == Action.DOWNLEFT:
        if prev_action == Action.DOWN:
            return Action.LEFT
        else:
            return Action.DOWN
    else:
        return action


def dof(pos: chex.Array, maze: chex.Array):
    """
    Degree of freedom of the object, can it move up, right, left, down
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    TO BE PASSED AT LA MOULINETTE

























    """
    x, y = pos
    grid_x = (x+5)//4
    grid_y = (y+3)//4
    no_wall_above = sum(maze[grid_y-2, grid_x-1:grid_x+2]) == 0
    no_wall_bellow = sum(maze[grid_y+2, grid_x-1:grid_x+2]) == 0
    no_wall_left = sum(maze[grid_y-1:grid_y+2, grid_x-2]) == 0
    no_wall_right = sum(maze[grid_y-1:grid_y+2, grid_x+2]) == 0
    # if x % 4 == 1: # can potentially move up/down
    # if y % 12 == 6: # can potentially move left/right
    return no_wall_above, no_wall_right, no_wall_left, no_wall_bellow


def can_change_direction(pos: chex.Array, maze: chex.Array):
    """
    Wether the object change change direction
    """
    up, right, left, down = dof(pos, maze)
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    return up and on_vertical_grid, right and on_horizontal_grid, left and on_horizontal_grid, down and on_vertical_grid


def stop_wall(pos: chex.Array, maze: chex.Array):
    up, right, left, down = dof(pos, maze)
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    return not(up) and on_horizontal_grid, not(right) and on_vertical_grid, not(left) and on_vertical_grid, not(down) and on_horizontal_grid



class PacmanState(NamedTuple):
    pacman_pos: chex.Array  # (x, y)
    pacman_dir: chex.Array  # (dx, dy)
    current_action: chex.Array # 0: NOOP, 1: NOOP, 2: UP ...
    ghost_positions: chex.Array  # (N_ghosts, 2)
    ghost_dirs: chex.Array  # (N_ghosts, 2)
    # pellets: chex.Array  # 2D grid of 0 (empty) or 1 (pellet)
    # power_pellets: chex.Array
    score: chex.Array
    step_count: chex.Array
    game_over: chex.Array
    # power_mode_timer: chex.Array
    level: chex.Array
    lives: chex.Array  # Number of lives left
    death_timer: chex.Array  # Frames left in death animation

class PacmanObservation(NamedTuple):
    grid: chex.Array  # 2D array showing layout of walls, pellets, pacman, ghosts

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array

# Example directions
DIRECTIONS = jnp.array([
    [0, 0],   # NOOP
    [0, 0],   # FIRE
    [0, -1],  # UP
    [1, 0],   # RIGHT
    [-1, 0],  # LEFT
    [0, 1],   # DOWN
])

class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo]):
    def __init__(self):
        super().__init__()
        self.frame_stack_size = 1
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
        ]
        self.maze_layout = MAZES[BASE_LEVEL]
    
    def action_space(self) -> spaces.Discrete:
        """Returns the action space for MsPacman.
        Actions are:
        0: NOOP
        1: FIRE
        2: UP
        3: RIGHT
        4: LEFT
        5: DOWN
        6: UPRIGHT
        7: UPLEFT
        8: DOWNRIGHT
        9: DOWNLEFT
        """
        return spaces.Discrete(10)

    def reset(self, key=None) -> Tuple[PacmanObservation, PacmanState]:
        pacman_pos = jnp.array([75, 102])
        pacman_dir = jnp.array([-1, 0])
        ghost_positions = jnp.array([[40, 78], [50, 78], [75, 54], [120, 78]])
        ghost_dirs = jnp.zeros_like(ghost_positions)
        pellets = (self.maze_layout == 0).astype(jnp.int32)
        power_pellets = jnp.zeros_like(pellets)
        power_pellets = power_pellets.at[1, 1].set(1)
        power_pellets = power_pellets.at[1, 17].set(1)
        power_pellets = power_pellets.at[9, 1].set(1)
        power_pellets = power_pellets.at[9, 17].set(1)
        pellets = pellets - power_pellets

        power_mode_timer = jnp.array(0)

        state = PacmanState(
            pacman_pos=pacman_pos,
            pacman_dir=pacman_dir,
            current_action = 4,
            ghost_positions=ghost_positions,
            ghost_dirs=ghost_dirs,
            # pellets=pellets,
            # power_pellets=power_pellets,
            score=jnp.array(0),
            step_count=jnp.array(0),
            game_over=jnp.array(False),
            # power_mode_timer=power_mode_timer,
            level=0,
            lives=jnp.array(3),
            death_timer=jnp.array(0),
        )
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array, key: chex.PRNGKey) -> tuple[
        PacmanObservation, PacmanState, Array, Array, PacmanInfo]:
        # If in death animation, decrement timer and freeze everything
        if state.death_timer > 0:
            new_death_timer = state.death_timer - 1
            # When timer reaches 0, reset positions if lives remain
            if new_death_timer == 0 and state.lives > 0:
                pacman_pos = jnp.array([75, 102])
                pacman_dir = jnp.array([-1, 0])
                ghost_positions = jnp.array([[40, 78], [50, 78], [75, 54], [120, 78]])
                ghost_dirs = jnp.zeros_like(ghost_positions)
            else:
                pacman_pos = state.pacman_pos
                pacman_dir = state.pacman_dir
                ghost_positions = state.ghost_positions
                ghost_dirs = state.ghost_dirs
            game_over = (state.lives == 0) & (new_death_timer == 0)
            new_state = PacmanState(
                pacman_pos=pacman_pos,
                pacman_dir=pacman_dir,
                current_action=state.current_action,
                ghost_positions=ghost_positions,
                ghost_dirs=ghost_dirs,
                score=state.score,
                step_count=state.step_count + 1,
                game_over=game_over,
                level=state.level,
                lives=state.lives,
                death_timer=new_death_timer,
            )
            obs = self._get_observation(new_state)
            reward = 0.0
            done = game_over
            info = PacmanInfo(score=state.score, done=done)
            return obs, new_state, reward, done, info

        action = last_pressed_action(action, state.current_action)
        possible_directions = can_change_direction(state.pacman_pos, self.maze_layout)
        if action != Action.NOOP and action != Action.FIRE and possible_directions[action - 2]:
            new_pacman_dir = DIRECTIONS[action]
            executed_action = action
        else:
            # check for wall collision
            if state.current_action > 1 and stop_wall(state.pacman_pos, self.maze_layout)[state.current_action - 2]:
                executed_action = 0
                new_pacman_dir = jnp.array([0, 0])
            else:
                executed_action = state.current_action
                new_pacman_dir = state.pacman_dir
        if state.step_count % 2:
            new_pacman_pos = state.pacman_pos + new_pacman_dir
            new_pacman_pos = new_pacman_pos.at[0].set(new_pacman_pos[0] % 160)
        else:
            new_pacman_pos = state.pacman_pos



        # has_pellet = state.pellets[new_pacman_pos[1], new_pacman_pos[0]] > 0
        # has_power = state.power_pellets[new_pacman_pos[1], new_pacman_pos[0]] > 0

        # # Consume pellet
        # pellets = state.pellets.at[new_pacman_pos[1], new_pacman_pos[0]].set(0)
        # power_pellets = state.power_pellets.at[new_pacman_pos[1], new_pacman_pos[0]].set(0)
        score = state.score #+ jax.lax.select(has_pellet, 10, 0)

        #Update power mode timer
        # power_mode_timer = jax.lax.select(
        #     has_power,
        #     jnp.array(50), #Reset timer when power pellet is consumed
        #     jnp.maximum(0, state.power_mode_timer - 1)
        # )

        # Ghost random movement
        # def move_one_ghost(ghost_pos, key):
        #     return ghost_frightened_step(ghost_pos, self._get_wall_grid(), key)

        # keys = random.split(random.PRNGKey(state.step_count), state.ghost_positions.shape[0])
        # ghost_positions = jax.vmap(move_one_ghost)(state.ghost_positions, keys)
        ghost_positions, ghost_dirs = ghosts_step(
            state.ghost_positions, state.ghost_dirs, self.maze_layout, key=key
        )
        # Collision detection
        collision = jnp.any(jnp.all(abs(new_pacman_pos - ghost_positions) < 8, axis=1))
        new_lives = state.lives - jnp.where(collision, 1, 0)
        new_death_timer = jnp.where(collision, 40, 0)
        game_over = (new_lives == 0) & (new_death_timer > 0)
        new_state = PacmanState(
            pacman_pos=new_pacman_pos,
            pacman_dir=new_pacman_dir,
            current_action=executed_action,
            ghost_positions=ghost_positions,
            ghost_dirs=ghost_dirs,
            score=score,
            step_count=state.step_count + 1,
            game_over=game_over,
            level=state.level,
            lives=new_lives,
            death_timer=new_death_timer,
        )
        obs = self._get_observation(new_state)
        reward = 0.0
        done = game_over
        info = PacmanInfo(score=score, done=done)
        return obs, new_state, reward, done, info

    def _get_wall_grid(self):
        return self.maze_layout

    def _get_observation(self, state: PacmanState) -> PacmanObservation:
        grid = self.maze_layout.copy()
        return grid



class MsPacmanRenderer(AtraJaxisRenderer):
    """JAX-based MsPacman game renderer, optimized with JIT compilation."""

    def __init__(self):
        super().__init__()
        self.SPRITE_BG = load_background(BASE_LEVEL)
        self.SPRITE_PLAYER = pacman_rgba()
        self.SPRITE_GHOSTS = load_ghosts()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.SPRITE_BG
        raster = aj.render_at(raster, state.pacman_pos[0], state.pacman_pos[1], self.SPRITE_PLAYER)
        for g_pos, g_sprite in zip(state.ghost_positions, self.SPRITE_GHOSTS):
            raster = aj.render_at(raster, g_pos[0], g_pos[1], g_sprite) 
        return raster


def get_direction_index(direction: chex.Array) -> int:
    """
    Returns the index in DIRECTIONS for a given direction vector.
    """
    for idx, d in enumerate(DIRECTIONS):
        if jnp.all(d == direction):
            return idx
    return 0  # Default to NOOP if not found


def ghost_step(ghost_pos: chex.Array, ghost_dir: chex.Array, maze: chex.Array, key=None) -> Tuple[chex.Array, chex.Array]:
    """
    Step function for a single ghost. Never stops, never reverses, can change direction at intersections.
    """
    possible = can_change_direction(ghost_pos, maze)
    dir_idx = get_direction_index(ghost_dir)
    # Map: 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN
    direction_indices = [2, 3, 4, 5]
    # Opposite directions: UP<->DOWN, LEFT<->RIGHT
    opposite = {2:5, 3:4, 4:3, 5:2}
    # Build list of allowed directions (not reverse, not blocked)
    allowed = []
    for i, can_go in zip(direction_indices, possible):
        if can_go and (dir_idx == 0 or i != opposite.get(dir_idx, -1)):
            allowed.append(i)
    if not allowed:
        # If no allowed (shouldn't happen), keep going forward
        next_dir_idx = dir_idx
    elif len(allowed) == 1:
        next_dir_idx = allowed[0]
    else:
        # Randomly pick one (except reverse)
        if key is not None:
            next_dir_idx = jax.random.choice(key, jnp.array(allowed))
        else:
            next_dir_idx = allowed[0]  # deterministic fallback
    next_dir = DIRECTIONS[next_dir_idx]
    new_pos = ghost_pos + next_dir
    new_pos = new_pos.at[0].set(new_pos[0] % 160)  # wrap horizontally
    return new_pos, next_dir


def ghosts_step(ghost_positions: chex.Array, ghost_dirs: chex.Array, maze: chex.Array, key=None) -> Tuple[chex.Array, chex.Array]:
    """
    Step all ghosts. key can be a PRNGKey or None for deterministic.
    """
    n_ghosts = ghost_positions.shape[0]
    if key is not None:
        keys = jax.random.split(key, n_ghosts)
    else:
        keys = [None] * n_ghosts
    new_positions = []
    new_dirs = []
    for i in range(n_ghosts):
        pos, d = ghost_step(ghost_positions[i], ghost_dirs[i], maze, keys[i])
        new_positions.append(pos)
        new_dirs.append(d)
    return jnp.stack(new_positions), jnp.stack(new_dirs)
