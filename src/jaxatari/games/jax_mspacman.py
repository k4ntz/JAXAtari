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
from jaxatari.games.mspacman_mazes import MAZES, load_background, pacmans_rgba, load_ghosts, precompute_dof, base_pellets

from jax import random, Array



WIDTH = 160
HEIGHT = 210

RESET_LEVEL = 0
POWER_PELLET_DURATION = 62  # Duration of power pellet effect in frames (x8 steps)
ppx0 = 8
ppx1 = 148
ppy0 = 18
ppy1 = 150
EATEN_GHOST_DURATION = 120 # in steps
POWER_PELLET_POSITIONS = [[ppx0, ppy0], [ppx1, ppy0], [ppx0, ppy1], [ppx1, ppy1]]
INITIAL_GHOSTS_POSITIONS = jnp.array([[40, 78], [50, 78], [75, 54], [120, 78]])
PELLETS_TO_COLLECT = 155  # Total pellets to collect in the maze (including power pellets)
# PELLETS_TO_COLLECT = 5  # Total pellets to collect in the maze (including power pellets)
PELLET_POINTS = 10
POWER_PELLET_POINTS = 50
# cherry, strawberry, orange, pretzel, apple, pear, banana
FRUITS_POINTS = [100, 200, 500, 700, 1000, 2000, 5000]  
GHOSTS_POINTS = [200, 400, 800, 1600]
BONUS_LIFE_LIMIT = 10000


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


def dof(pos: chex.Array, dofmaze: chex.Array):
    """
    Degree of freedom of the object, can it move up, right, left, down
    """
    x, y = pos
    grid_x = (x+5)//4
    grid_y = (y+3)//4
    return dofmaze[grid_x][grid_y]


def available_directions(pos: chex.Array, dofmaze: chex.Array):
    """
    Wether the object change change direction
    """
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    up, right, left, down = dof(pos, dofmaze)
    return up and on_vertical_grid, right and on_horizontal_grid, left and on_horizontal_grid, down and on_vertical_grid


def stop_wall(pos: chex.Array, dofmaze: chex.Array):
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    up, right, left, down = dof(pos, dofmaze)
    return not(up) and on_horizontal_grid, not(right) and on_vertical_grid, not(left) and on_vertical_grid, not(down) and on_horizontal_grid



class PacmanState(NamedTuple):
    pacman_pos: chex.Array  # (x, y)
    pacman_dir: chex.Array  # (dx, dy)
    pacman_last_dir_int : chex.Array  # Last direction as an integer (0: UP, 1: RIGHT, 2: LEFT, 3: DOWN)
    current_action: chex.Array # 0: NOOP, 1: NOOP, 2: UP ...
    ghost_positions: chex.Array  # (N_ghosts, 2)
    ghosts_dirs: chex.Array  # (N_ghosts, 2)
    pellets: chex.Array  # 2D grid of 0 (empty) or 1 (pellet)
    collected_pellets: chex.Array  # the number of pellets collected
    has_pellet: chex.Array  # Boolean indicating if pacman just collected a pellet
    power_pellets: chex.Array
    score: chex.Array
    step_count: chex.Array
    game_over: chex.Array
    power_mode_timer: chex.Array # Timer for power mode, decrements every 8 steps
    eaten_ghosts: chex.Array  # timers indicating which ghosts have been eaten, when set to one, does not go down, indicate respawned ghost
    level: chex.Array
    lives: chex.Array  # Number of lives left
    death_timer: chex.Array  # Frames left in death animation
    maze_layout: chex.Array # Whether the level is completed
    completed_level: chex.Array  # Whether the level is completed
    dofmaze: chex.Array # Precomputed degree of freedom maze layout


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
        pacman_pos = jnp.array([17, 126])

        state = PacmanState(
            pacman_pos=pacman_pos,
            pacman_dir=jnp.array([-1, 0]),
            pacman_last_dir_int=jnp.array(2),  # Default to LEFT
            current_action = 4,
            ghost_positions=jnp.array([[40, 78], [50, 78], [75, 54], [120, 78]]),
            ghosts_dirs=jnp.zeros((4, 2), dtype=jnp.int8),  # Ghosts start with no direction
            pellets=jnp.copy(base_pellets),
            collected_pellets=jnp.array(0).astype(jnp.uint8),
            has_pellet=jnp.array(False),
            power_pellets=jnp.ones(4, dtype=jnp.bool_),
            eaten_ghosts=jnp.zeros(4, dtype=jnp.int32),  # Timers for eaten ghosts
            score=jnp.array(0),
            step_count=jnp.array(0),
            game_over=jnp.array(False),
            power_mode_timer=jnp.array(0).astype(jnp.uint8),  # Timer for power mode,
            level=0,
            lives=jnp.array(3),
            death_timer=jnp.array(0),
            completed_level=jnp.array(False),
            maze_layout=RESET_LEVEL,
            dofmaze=precompute_dof(MAZES[RESET_LEVEL]) # Precompute degree of freedom maze layout
        )
        obs = None
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array, key: chex.PRNGKey) -> tuple[
        PacmanObservation, PacmanState, Array, Array, PacmanInfo]:
        # If in death animation, decrement timer and freeze everything
        eaten_ghosts = state.eaten_ghosts
        power_mode_timer = state.power_mode_timer
        completed_level = False
        pacman_last_dir_int = state.pacman_last_dir_int
        dofmaze = state.dofmaze
        if state.death_timer > 0:
            new_death_timer = state.death_timer - 1
            # When timer reaches 0, reset positions if lives remain
            if new_death_timer == 0 and state.lives > 0:
                eaten_ghosts = jnp.zeros(4, dtype=jnp.int32)  # Reset eaten ghosts
                power_mode_timer = jnp.array(0).astype(jnp.uint8)  # Reset power mode timer
                pacman_pos = jnp.array([75, 102])
                pacman_dir = jnp.array([-1, 0])
                ghost_positions = INITIAL_GHOSTS_POSITIONS
                ghosts_dirs = jnp.zeros_like(ghost_positions)
            else:
                pacman_pos = state.pacman_pos
                pacman_dir = state.pacman_dir
                ghost_positions = state.ghost_positions
                ghosts_dirs = state.ghosts_dirs
            game_over = (state.lives == 0) & (new_death_timer == 0)
            new_state = PacmanState(
                pacman_pos=pacman_pos,
                pacman_dir=pacman_dir,
                pacman_last_dir_int=jnp.array(2),
                current_action=state.current_action,
                ghost_positions=ghost_positions,
                ghosts_dirs=ghosts_dirs,
                pellets=state.pellets,
                collected_pellets=state.collected_pellets,
                has_pellet=state.has_pellet,
                power_pellets=state.power_pellets,
                eaten_ghosts=eaten_ghosts,
                power_mode_timer=power_mode_timer,
                score=state.score,
                step_count=state.step_count + 1,
                game_over=game_over,
                level=state.level,
                lives=state.lives,
                death_timer=new_death_timer,
                completed_level=completed_level,
                maze_layout=state.maze_layout, 
                dofmaze=dofmaze
            )
            obs = self._get_observation(new_state)
            reward = 0.0
            done = game_over
            info = PacmanInfo(score=state.score, done=done)
            return obs, new_state, reward, done, info



        action = last_pressed_action(action, state.current_action)
        possible_directions = available_directions(state.pacman_pos, state.dofmaze)
        if action != Action.NOOP and action != Action.FIRE and possible_directions[action - 2]:
            new_pacman_dir = DIRECTIONS[action]
            executed_action = action
            pacman_last_dir_int = action - 2  # Convert action to direction index (0: UP, 1: RIGHT, 2: LEFT, 3: DOWN)
        else:
            # check for wall collision
            if state.current_action > 1 and stop_wall(state.pacman_pos, dofmaze)[state.current_action - 2]:
                executed_action = 0
                new_pacman_dir = jnp.array([0, 0])
            else:
                executed_action = state.current_action
                new_pacman_dir = state.pacman_dir
        # if state.step_count % 2:
        if True:
            new_pacman_pos = state.pacman_pos + new_pacman_dir
            new_pacman_pos = new_pacman_pos.at[0].set(new_pacman_pos[0] % 160)
        else:
            new_pacman_pos = state.pacman_pos

        # power pellets 
        collected_pellets = state.collected_pellets
        power_pellets = state.power_pellets
        px, py = new_pacman_pos // 4
        ate_power_pill = False
        ghosts_dirs = state.ghosts_dirs
        if px == 1:
            if py == 3 or py == 4:
                power_pellets = state.power_pellets.at[0].set(False)
                ate_power_pill = True
            elif py == 36 or py == 37:
                power_pellets = state.power_pellets.at[2].set(False)
                ate_power_pill = True
        elif px == 36:
            if py == 3 or py == 4:
                power_pellets = state.power_pellets.at[1].set(False)
                ate_power_pill = True
            elif py == 36 or py == 37:
                power_pellets = state.power_pellets.at[3].set(False)
                ate_power_pill = True
        if ate_power_pill and state.power_mode_timer == 0:
            power_mode_timer = POWER_PELLET_DURATION
            collected_pellets += 1
            ghosts_dirs = jnp.array([-gh for gh in ghosts_dirs])  # Invert ghost directions
        if state.power_mode_timer > 0 and state.step_count % 8 == 0:
            # Decrement power mode timer
            power_mode_timer = state.power_mode_timer - 1
        pellets = state.pellets
        # Check for pellet consumption
        has_pellet = jnp.array(False)
        x_offset = 5 if new_pacman_pos[0] < 75 else 1
        if new_pacman_pos[0] % 8 == x_offset and new_pacman_pos[1] % 12 == 6:
            x_pellets = (new_pacman_pos[0] - 2) // 8
            y_pellets = (new_pacman_pos[1] + 4) // 12
            if state.pellets[x_pellets, y_pellets]:
                has_pellet = jnp.array(True)
                pellets = state.pellets.at[x_pellets, y_pellets].set(False)
        score = state.score + jax.lax.select(has_pellet, 10, 0)
        if has_pellet:
            collected_pellets = collected_pellets + 1

        ghost_positions, ghosts_dirs, eaten_ghosts = ghosts_step(
            state.ghost_positions, ghosts_dirs, dofmaze, 
            eaten_ghosts, power_mode_timer, key=key
        )
        # Collision detection
        if power_mode_timer > 0:
            # In power mode, ghosts are vulnerable
            collision = False
            for i in range(4): 
                if jnp.all(abs(new_pacman_pos - ghost_positions[i]) < 8):
                    if eaten_ghosts[i] == 1:  # If ghost already eaten and respawned
                        collision = True  # Collision with an already eaten and respawned ghost
                    else:
                        # Ghost eaten
                        score += 200
                        ghost_positions = ghost_positions.at[i].set((76, 70))  # Reset eaten ghost position
                        ghosts_dirs = ghosts_dirs.at[i].set([0, 0])  # Reset eaten ghost direction
                        eaten_ghosts = state.eaten_ghosts.at[i].set(EATEN_GHOST_DURATION)  # Set eaten ghost timer
        else:
            collision = jnp.any(jnp.all(abs(new_pacman_pos - ghost_positions) < 8, axis=1))
        new_lives = state.lives - jnp.where(collision, 1, 0)
        new_death_timer = jnp.where(collision, 40, 0)
        game_over = (new_lives == 0) & (new_death_timer > 0)
        maze_layout = state.maze_layout
        if collected_pellets >= PELLETS_TO_COLLECT:
            # If all pellets collected, reset game
            collected_pellets = jnp.array(0, dtype=jnp.uint8)
            ghost_positions = INITIAL_GHOSTS_POSITIONS
            ghosts_dirs = jnp.zeros_like(ghost_positions)
            power_mode_timer = jnp.array(0, dtype=jnp.uint8)
            power_pellets = jnp.ones(4, dtype=jnp.bool_)
            completed_level = jnp.array(True)
            score += 500
            pellets = jnp.copy(base_pellets)  # Reset pellets
            maze_layout = (maze_layout + 1) % 4  # len(MAZES)
            print(f"Level completed! New level: {maze_layout}")
            dofmaze= precompute_dof(MAZES[maze_layout])

        new_state = PacmanState(
            pacman_pos=new_pacman_pos,
            pacman_dir=new_pacman_dir,
            pacman_last_dir_int=pacman_last_dir_int,
            current_action=executed_action,
            ghost_positions=ghost_positions,
            ghosts_dirs=ghosts_dirs,
            pellets=pellets,
            collected_pellets=collected_pellets,
            has_pellet=has_pellet,
            power_pellets=power_pellets,
            power_mode_timer=power_mode_timer,
            eaten_ghosts=eaten_ghosts,
            score=score,
            step_count=state.step_count + 1,
            game_over=game_over,
            level=state.level,
            lives=new_lives,
            death_timer=new_death_timer,
            completed_level=completed_level,
            maze_layout=maze_layout,
            dofmaze=dofmaze,
        )
        obs = None
        reward = 0.0
        done = game_over
        info = PacmanInfo(score=score, done=done)
        return obs, new_state, reward, done, info



class MsPacmanRenderer(AtraJaxisRenderer):
    """JAX-based MsPacman game renderer, optimized with JIT compilation."""

    def __init__(self):
        super().__init__()
        self.SPRITE_BG = load_background(RESET_LEVEL)
        self.SPRITES_PLAYER = pacmans_rgba()
        self.SPRITES_GHOSTS = load_ghosts()
        self.power_pellet_sprite = jnp.ones((4, 7, 4), dtype=jnp.uint8) * 200

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        if state.completed_level:
            self.SPRITE_BG = load_background(state.maze_layout)
        raster = self.SPRITE_BG
        # de-render pellets
        # if state.has_pellet:
        if state.has_pellet:
            # PATH_COLOR = jnp.array([200, 200, 200], dtype=jnp.uint8)
            PATH_COLOR = jnp.array([0, 28, 136], dtype=jnp.uint8)
            pellet_x = state.pacman_pos[0] + 3
            pellet_y = state.pacman_pos[1] + 4
            for i in range(4):
                for j in range(2):
                    self.SPRITE_BG = self.SPRITE_BG.at[pellet_x+i, pellet_y+j].set(PATH_COLOR)
        # power pellets
        for i in range(4):
            if state.power_pellets[i]:
                pellet_x, pellet_y = POWER_PELLET_POSITIONS[i]
                raster = aj.render_at(raster, pellet_x, pellet_y, self.power_pellet_sprite)
        # 0: down, 1: left, 2: Nothing, 3: right, 4: up
        # orientation = state.pacman_dir[0] + 2 * (state.pacman_dir[1] + 1) 
        orientation = state.pacman_last_dir_int
        pacman_sprite = self.SPRITES_PLAYER[orientation][(state.step_count % 16) // 4]
        raster = aj.render_at(raster, state.pacman_pos[0], state.pacman_pos[1], 
                              pacman_sprite)
        ghosts_orientation = (state.step_count % 32) // 16
        for i, g_pos in enumerate(state.ghost_positions):
            if state.power_mode_timer > 0:
                # Render frightened ghost
                if state.eaten_ghosts[i] > 0:
                    g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][i]
                elif state.power_mode_timer < 10 and state.power_mode_timer % 2:
                    g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][5] # white blinking effect
                else:
                    g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][4] # blue ghost
            else:
                g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][i]
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

def get_chase_target(ghost_id: int,
                     pacman_tile: chex.Array,
                     pacman_dir: chex.Array,
                     blinky_tile: chex.Array) -> chex.Array:
    """
    Compute the chase-mode target tile for each ghost by ID:
     0=Red (Blinky), 1=Pink (Pinky), 2=Blue (Inky), 3=Orange (Clyde)
    """
    if ghost_id == 0:
        # Blinky: target Pac-Man's current tile
        return pacman_tile
    elif ghost_id == 1:
        # Pinky: 4 tiles ahead of Pac-Man
        ahead = pacman_tile + 4 * pacman_dir
        return ahead
    elif ghost_id == 2:
        # Inky: vector from Blinky to two tiles ahead of Pac-Man, doubled
        two_ahead = pacman_tile + 2 * pacman_dir
        vect = two_ahead - blinky_tile
        return blinky_tile + 2 * vect
    else:
        # Clyde: if >8 tiles away, chase Pac-Man, else scatter corner
        dist = jnp.linalg.norm(ghost_id_tile - pacman_tile)
        chase = pacman_tile
        corner = SCATTER_TARGETS[3]
        return jnp.where(dist > 8, chase, corner)


def ghost_step(ghost_pos: chex.Array, ghost_dir: chex.Array, dofmaze:chex.Array, 
                eaten_ghost: chex.Array, power_mode: chex.Array,
                key=None) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Step function for a single ghost. Never stops, never reverses, can change direction at intersections.
    """
    x, y = ghost_pos
    if x % 4 == 1 or y % 12 == 6: # on horizontal or vertical grid
        possible = available_directions(ghost_pos, dofmaze)
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
    else:
        next_dir = ghost_dir
    new_pos = ghost_pos + next_dir
    new_pos = new_pos.at[0].set(new_pos[0] % 160)  # wrap horizontally
    if eaten_ghost == 1:
        next_dir = jnp.array([0, -1])  # Reset direction to escape the center box
    if eaten_ghost > 1:
        eaten_ghost = eaten_ghost - 1
    elif power_mode == 0: # reinit ghosts, not eaten
        eaten_ghost = 0
    return new_pos, next_dir, eaten_ghost


def ghosts_step(ghost_positions: chex.Array, ghosts_dirs: chex.Array, dofmaze: chex.Array, 
                eaten_ghosts: chex.Array, power_mode: bool = chex.Array,
                key=None) -> Tuple[chex.Array, chex.Array, chex.Array]:
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
    eats = []
    for i in range(n_ghosts):
        pos, d, eat = ghost_step(ghost_positions[i], ghosts_dirs[i], dofmaze, eaten_ghosts[i], power_mode, keys[i])
        new_positions.append(pos)
        new_dirs.append(d)
        eats.append(eat)
    return jnp.stack(new_positions), jnp.stack(new_dirs), jnp.array(eats)
