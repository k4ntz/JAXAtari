"""
Group: Sooraj Rathore, Kadir Özen
Edit: Jan Rafflewski
"""

"""
TODO
1) Fruits
    1.1) Fruit spawning
    1.2) Fruit movement
    1.3) Fruit scoring
    1.4) Fruit animation
2) Ghosts
    2.1) Ghost pathfinding
    2.2) Ghost movement
3) Game
    3.1) Life system
    3.2) Gameover state
    3.3) Pacman death animation
    3.4) Level progression
"""


# --- IMPORTS --- #
from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.games.mspacman_mazes import (
    MAZES, load_background, load_score_digits, pacmans_rgba, load_ghosts,
    precompute_dof, base_pellets)


# --- CONSTANTS --- #
# GENERAL
RESET_LEVEL = 0 # the starting level, loaded when reset is called
RESET_TIMER = 40  # Timer for resetting the game after death
MAX_SCORE_DIGITS = 6 # Number of digits to display in the score
PELLETS_TO_COLLECT = 155  # Total pellets to collect in the maze (including power pellets)
INITIAL_LIFES = 2
BONUS_LIFE_LIMIT = 10000

# GHOST MODES
GHOST_RANDOM = 0
GHOST_CHASING = 1
GHOST_FRIGHTENED = 2
GHOST_BLINKING = 3
GHOST_RETURNING = 4
GHOST_ENJAILED = 5

# GHOST TIMINGS
FRIGHTENED_DURATION = 62*8 # Duration of power pellet effect in frames (x8 steps)
BLINKING_DURATION = 10*8
GHOST_JAIL_DURATION = 120 # in steps

# POSITIONS
PPX0 = 8
PPX1 = 148
PPY0 = 20
PPY1 = 152
POWER_PELLET_POSITIONS = [[PPX0, PPY0], [PPX1, PPY0], [PPX0, PPY1], [PPX1, PPY1]]
INITIAL_GHOSTS_POSITIONS = jnp.array([[40, 78], [50, 78], [75, 54], [120, 78]])

# DIRECTIONS
DIRECTIONS = jnp.array([
    [0, 0],   # NOOP
    [0, 0],   # FIRE
    [0, -1],  # UP
    [1, 0],   # RIGHT
    [-1, 0],  # LEFT
    [0, 1],   # DOWN
])

# POINTS
PELLET_POINTS = 10
POWER_PELLET_POINTS = 50
FRUITS_POINTS = [100, 200, 500, 700, 1000, 2000, 5000] # cherry, strawberry, orange, pretzel, apple, pear, banana
EAT_GHOSTS_BASE_POINTS = 200

# COLORS
PATH_COLOR = jnp.array([0, 28, 136], dtype=jnp.uint8)
WALL_COLOR = jnp.array([228, 111, 111], dtype=jnp.uint8)
PELLET_COLOR = WALL_COLOR  # Same color as walls for pellets
POWER_PELLET_SPRITE = jnp.tile(jnp.concatenate([PELLET_COLOR, jnp.array([255], dtype=jnp.uint8)]), (4, 7, 1))  # 4x7 sprite 


# --- CLASSES --- #
class PacmanState(NamedTuple):
    pacman_pos: chex.Array  # (x, y)
    pacman_dir: chex.Array  # (dx, dy)
    pacman_last_dir_int : chex.Array  # Last direction as an integer (0: UP, 1: RIGHT, 2: LEFT, 3: DOWN)
    current_action: chex.Array # 0: NOOP, 1: NOOP, 2: UP ...
    ghost_positions: chex.Array  # (N_ghosts, 2)
    ghosts_dirs: chex.Array  # (N_ghosts, 2)
    ghosts_modes: chex.Array  # (N_ghosts)
    ghosts_timers: chex.Array  # (N_ghosts)
    eaten_ghosts: chex.Array
    pellets: chex.Array  # 2D grid of 0 (empty) or 1 (pellet)
    collected_pellets: chex.Array  # the number of pellets collected
    has_pellet: chex.Array  # Boolean indicating if pacman just collected a pellet
    power_pellets: chex.Array
    score: chex.Array
    score_changed: chex.Array
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
    reset: chex.Array  # Reset state for the next episode


class PacmanObservation(NamedTuple):
    grid: chex.Array  # 2D array showing layout of walls, pellets, pacman, ghosts


class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array


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
            ghosts_modes=jnp.zeros(4),
            ghosts_timers=jnp.zeros(4),
            eaten_ghosts=jnp.array(0).astype(jnp.uint8), # number of eaten ghost since power pellet consumed
            pellets=jnp.copy(base_pellets),
            collected_pellets=jnp.array(0).astype(jnp.uint8),
            has_pellet=jnp.array(False),
            power_pellets=jnp.ones(4, dtype=jnp.bool_),
            score=jnp.array(0),
            score_changed=jnp.zeros(MAX_SCORE_DIGITS, dtype=jnp.bool_), # indicates which score digit changed since the last step
            step_count=jnp.array(0),
            game_over=jnp.array(False),
            power_mode_timer=jnp.array(0).astype(jnp.uint8),  # Timer for power mode,
            level=0,
            lives=jnp.array(INITIAL_LIFES, dtype=jnp.int8),  # Number of lives left
            death_timer=jnp.array(0),
            completed_level=jnp.array(False),
            maze_layout=RESET_LEVEL,
            dofmaze=precompute_dof(MAZES[RESET_LEVEL]), # Precompute degree of freedom maze layout
            reset=jnp.array(True, dtype=jnp.bool_) # to reload the background
        )
        obs = None
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array, key: chex.PRNGKey) -> tuple[
        PacmanObservation, PacmanState, jax.Array, jax.Array, PacmanInfo]:
        # If in death animation, decrement timer and freeze everything
        power_mode_timer = state.power_mode_timer
        completed_level = False
        pacman_last_dir_int = state.pacman_last_dir_int
        dofmaze = state.dofmaze
        if state.death_timer > 0:
            new_death_timer = state.death_timer - 1
            # When timer reaches 0, reset positions if lives remain
            if new_death_timer == 0 and state.lives > 0:
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
                ghosts_modes=jnp.zeros(4),
                ghosts_timers=jnp.zeros(4),
                eaten_ghosts=state.eaten_ghosts,
                pellets=state.pellets,
                collected_pellets=state.collected_pellets,
                has_pellet=state.has_pellet,
                power_pellets=state.power_pellets,
                power_mode_timer=power_mode_timer,
                score=state.score,
                score_changed=jnp.zeros(MAX_SCORE_DIGITS, dtype=jnp.bool_),
                step_count=state.step_count + 1,
                game_over=game_over,
                level=state.level,
                lives=state.lives,
                death_timer=new_death_timer,
                completed_level=completed_level,
                maze_layout=state.maze_layout, 
                dofmaze=dofmaze,
                reset=jnp.array(False, dtype=jnp.bool_)
            )
            obs = None
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
            # check for wall deadly_collision
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
            if (py == 3 or py == 4) and power_pellets[0]:
                power_pellets = state.power_pellets.at[0].set(False)
                ate_power_pill = True
            elif (py == 36 or py == 37) and power_pellets[2]:
                power_pellets = state.power_pellets.at[2].set(False)
                ate_power_pill = True
        elif px == 36:
            if (py == 3 or py == 4) and power_pellets[1]:
                power_pellets = state.power_pellets.at[1].set(False)
                ate_power_pill = True
            elif (py == 36 or py == 37) and power_pellets[3]:
                power_pellets = state.power_pellets.at[3].set(False)
                ate_power_pill = True
        if ate_power_pill:
            collected_pellets += 1
        elif state.power_mode_timer > 0 and state.step_count % 8 == 0:
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
        score = state.score + jax.lax.select(has_pellet, PELLET_POINTS, 0)
        if has_pellet:
            print(f"Pacman collected a pellet at {new_pacman_pos}, score: {score}")
            collected_pellets = collected_pellets + 1

        ghost_positions, ghosts_dirs, ghosts_modes, ghosts_timers = ghosts_step(
            state.ghost_positions, ghosts_dirs, state.ghosts_modes, state.ghosts_timers, 
            ate_power_pill, dofmaze, key=key
        )
        # Collision detection
        eaten_ghosts = state.eaten_ghosts
        deadly_collision = False
        for i in range(4): 
            if jnp.all(abs(new_pacman_pos - ghost_positions[i]) < 8):
                if ghosts_modes[i] == GHOST_FRIGHTENED or ghosts_modes[i] == GHOST_BLINKING:  # If are frighted
                    # Ghost eaten
                    score += EAT_GHOSTS_BASE_POINTS * (2 ** eaten_ghosts)
                    ghost_positions = ghost_positions.at[i].set((76, 70))  # Reset eaten ghost position
                    ghosts_dirs = ghosts_dirs.at[i].set([0, 0])  # Reset eaten ghost direction
                    ghosts_modes = ghosts_modes.at[i].set(GHOST_ENJAILED)
                    ghosts_timers = ghosts_timers.at[i].set(GHOST_JAIL_DURATION)
                    eaten_ghosts = eaten_ghosts + 1
                else:
                    deadly_collision = True  # Collision with an already eaten and respawned ghost
                    
        else:
            deadly_collision = jnp.any(jnp.all(abs(new_pacman_pos - ghost_positions) < 8, axis=1))
        new_lives = state.lives - jnp.where(deadly_collision, 1, 0)
        new_death_timer = jnp.where(deadly_collision, RESET_TIMER, 0)
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

        # Flag score change digit-wise
        if score != state.score:
            # score_str       = str(score)
            # state_score_str = str(state.score)
            # max_len         = max(len(score_str), len(state_score_str))
            # score_str       = score_str.zfill(max_len)
            # state_score_str = state_score_str.zfill(max_len)
            score_digits        = aj.int_to_digits(score, max_digits=MAX_SCORE_DIGITS)
            state_score_digits  = aj.int_to_digits(state.score, max_digits=MAX_SCORE_DIGITS)
            score_changed       = score_digits != state_score_digits
        else:
            score_changed       = jnp.array(False, dtype=jnp.bool_)

        new_state = PacmanState(
            pacman_pos=new_pacman_pos,
            pacman_dir=new_pacman_dir,
            pacman_last_dir_int=pacman_last_dir_int,
            current_action=executed_action,
            ghost_positions=ghost_positions,
            ghosts_dirs=ghosts_dirs,
            ghosts_modes=ghosts_modes,
            ghosts_timers=ghosts_timers,
            eaten_ghosts=eaten_ghosts,
            pellets=pellets,
            collected_pellets=collected_pellets,
            has_pellet=has_pellet,
            power_pellets=power_pellets,
            power_mode_timer=power_mode_timer,
            score=score,
            score_changed=score_changed,
            step_count=state.step_count + 1,
            game_over=game_over,
            level=state.level,
            lives=new_lives,
            death_timer=new_death_timer,
            completed_level=completed_level,
            maze_layout=maze_layout,
            dofmaze=dofmaze,
            reset=jnp.array(False, dtype=jnp.bool_)
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
        self.SPRITES_PLAYER = pacmans_rgba()
        self.SPRITES_GHOSTS = load_ghosts()
        self.SPRITES_DIGITS = load_score_digits()
        # self.reset_bg()

    def reset_bg(self):
        """Reset the background for a new level."""
        life_sprite = self.SPRITES_PLAYER[1][1] # Life sprite (right looking pacman)
        self.SPRITE_BG = load_background(RESET_LEVEL)
        self.SPRITE_BG = render_score(self.SPRITE_BG, 0, jnp.eye(1, MAX_SCORE_DIGITS, MAX_SCORE_DIGITS-1, dtype=jnp.bool_).ravel(), self.SPRITES_DIGITS)
        self.SPRITE_BG = render_lives(self.SPRITE_BG, INITIAL_LIFES, life_sprite)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        if state.reset:
            # Render game over screen
            self.reset_bg()
        if state.completed_level:
            self.SPRITE_BG = load_background(state.maze_layout)
        raster = self.SPRITE_BG
        # de-render pellets
        # if state.has_pellet:
        if state.has_pellet:
            pellet_x = state.pacman_pos[0] + 3
            pellet_y = state.pacman_pos[1] + 4
            for i in range(4):
                for j in range(2):
                    self.SPRITE_BG = self.SPRITE_BG.at[pellet_x+i, pellet_y+j].set(PATH_COLOR)
        # power pellets
        for i in range(2):
            pel_n = 2*i + ((state.step_count & 0b1000) >> 3) # Alternate power pellet rendering
            if state.power_pellets[pel_n]:
                pellet_x, pellet_y = POWER_PELLET_POSITIONS[pel_n]
                raster = aj.render_at(raster, pellet_x, pellet_y, POWER_PELLET_SPRITE)
        orientation = state.pacman_last_dir_int
        pacman_sprite = self.SPRITES_PLAYER[orientation][((state.step_count & 0b1000) >> 2)]
        raster = aj.render_at(raster, state.pacman_pos[0], state.pacman_pos[1], 
                              pacman_sprite)
        ghosts_orientation = ((state.step_count & 0b10000) >> 4) # (state.step_count % 32) // 16

        # Render score if changed
        if jnp.any(state.score_changed):
            self.SPRITE_BG = render_score(self.SPRITE_BG, state.score, state.score_changed, self.SPRITES_DIGITS)

        for i, g_pos in enumerate(state.ghost_positions):
            # Render frightened ghost
            if not (state.ghosts_modes[i] == GHOST_FRIGHTENED or state.ghosts_modes[i] == GHOST_BLINKING):
                g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][i]
            elif state.ghosts_modes[i] == GHOST_BLINKING and ((state.step_count & 0b1000) >> 3):
                g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][5] # white blinking effect
            else:
                g_sprite = self.SPRITES_GHOSTS[ghosts_orientation][4] # blue ghost
            raster = aj.render_at(raster, g_pos[0], g_pos[1], g_sprite)

        # Remove one life if a life is lost
        if state.death_timer == RESET_TIMER-1:
            self.SPRITE_BG = render_lives(self.SPRITE_BG, state.lives, self.SPRITES_PLAYER[1][1])
        return raster


# --- HELPER FUNCTIONS --- #
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
    What direction Pacman or the ghosts can take when at an intersection.
    Returns a tuple of booleans (up, right, left, down) indicating if
    the character can move in that direction.
    The character can only change direction if it is on a vertical or horizontal grid.

    Arguments:
    pos -- (x, y) position of the character
    dofmaze -- precomputed degree of freedom for a maze level/layout

    Returns:
    A tuple of booleans (up, right, left, down) indicating if the 
    character can move in that direction.
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


def ghosts_step(ghost_positions: chex.Array, ghosts_dirs: chex.Array, 
                ghosts_modes: chex.Array, ghosts_timers: chex.Array,
                ate_power_pill: chex.Array, dofmaze: chex.Array, key: chex.Array
                ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Step all ghosts. key can be a PRNGKey or None for deterministic.
    """
    n_ghosts = ghost_positions.shape[0]
    keys = jax.random.split(key, n_ghosts)
    new_positions = []
    new_dirs = []
    modes = []
    timers = []
    for i in range(n_ghosts):
        pos, dir, mode, timer = ghost_step(ghost_positions[i], ghosts_dirs[i], ghosts_modes[i], 
                                 ghosts_timers[i], ate_power_pill, dofmaze, keys[i])
        new_positions.append(pos)
        new_dirs.append(dir)
        modes.append(mode)
        timers.append(timer)
    return jnp.stack(new_positions), jnp.stack(new_dirs), jnp.array(modes), jnp.array(timers)


def ghost_step(ghost_pos: chex.Array, ghost_dir: chex.Array, 
                ghost_mode: chex.Array, ghost_timer: chex.Array,
                ate_power_pill: chex.Array, dofmaze: chex.Array, key: chex.Array
                ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Step function for a single ghost. Never stops, never reverses, can change direction at intersections.
    """
    if ate_power_pill and not(ghost_mode == GHOST_ENJAILED or ghost_mode == GHOST_RETURNING):
        return ghost_pos-ghost_dir, -ghost_dir, GHOST_FRIGHTENED, FRIGHTENED_DURATION-BLINKING_DURATION
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
    if ghost_mode == GHOST_ENJAILED and ghost_timer == 0:
        next_dir = jnp.array([0, -1])  # Reset direction to escape the center box
        ghost_mode = GHOST_RANDOM
    if ghost_mode == GHOST_FRIGHTENED and ghost_timer == 0:
        ghost_mode = GHOST_BLINKING
        ghost_timer = BLINKING_DURATION
    if ghost_mode == GHOST_BLINKING and ghost_timer == 0:
        ghost_mode = GHOST_RANDOM
    if ghost_timer > 0:
        ghost_timer = ghost_timer - 1
    return new_pos, next_dir, ghost_mode, ghost_timer


def render_score(raster, score, score_changed, digit_sprites, score_x=60, score_y=190, spacing=1, bg_color=jnp.array([0, 0, 0], dtype=jnp.uint8)):
    """
    Render the score on the raster at a fixed position.
    Only updates digits that have changed.
    """
    digits = aj.int_to_digits(score, max_digits=MAX_SCORE_DIGITS)
    for idx in range(MAX_SCORE_DIGITS):
        if score_changed[idx]:
            d_sprite    = digit_sprites[digits[idx]]
            bg_sprite   = jnp.full(d_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
            raster      = aj.render_at(raster, score_x + idx * (d_sprite.shape[1] + spacing), score_y, bg_sprite)
            raster      = aj.render_at(raster, score_x + idx * (d_sprite.shape[1] + spacing), score_y, d_sprite)
    return raster


def render_lives(raster, current_lifes, life_sprite, initial_lifes=INITIAL_LIFES, life_x=12, life_y=182, spacing=4, bg_color=jnp.array([0, 0, 0], dtype=jnp.uint8)):
    """
    Render the lives on the raster at a fixed position.
    """
    if current_lifes > initial_lifes:
        raise ValueError("Number of current lives cannot exceed the number of initial lives!")
    elif current_lifes < 0:
        raise ValueError("Number of current lives cannot be negative!")
    elif current_lifes == initial_lifes:
        for i in range(current_lifes):
            raster = aj.render_at(raster, life_x + i * (life_sprite.shape[1] + spacing), life_y, life_sprite)
    else:
        bg_sprite = jnp.full(life_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
        raster = aj.render_at(raster, life_x + current_lifes * (life_sprite.shape[1] + spacing), life_y, bg_sprite)
    return raster
