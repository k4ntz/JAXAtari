"""
Group: Sooraj Rathore, Kadir Özen
Edit: Jan Rafflewski
"""

"""
TODO
1) Fruits
    1.1) [ ] Fruit spawning
    1.2) [ ] Fruit movement
    1.3) [ ] Fruit scoring
    1.4) [ ] Fruit animation
2) Ghosts
    2.1) [x] Ghost pathfinding
    2.2) [x] Ghost movement
3) Game
    3.1) [x] Life system
    3.2) [x] Gameover state
    3.3) [ ] Level progression
    3.4) [ ] Correct speed
    3.5) [ ] Pacman death animation
"""


# --- IMPORTS --- #
from enum import IntEnum
import os
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.games.mspacman_mazes import MsPacmanMaze

# --- CONSTANTS --- #
# GENERAL
RESET_MAZE = 0 # the starting level, loaded when reset is called
RESET_TIMER = 40  # Timer for resetting the game after death
MAX_SCORE_DIGITS = 6 # Number of digits to display in the score
PELLETS_TO_COLLECT = 155  # Total pellets to collect in the maze (including power pellets)
INITIAL_LIVES = 2
BONUS_LIFE_LIMIT = 10000

# GHOST TYPES
class GhostType(IntEnum):
    BLINKY = 0
    PINKY = 1
    INKY = 2
    SUE = 3

# GHOST MODES
class GhostMode(IntEnum):
    RANDOM = 0
    CHASE = 1
    SCATTER = 2
    FRIGHTENED = 3
    BLINKING = 4
    RETURNING = 5
    ENJAILED = 6

# GHOST TIMINGS
CHASE_DURATION = 200*8 # Chosen randomly for now, should be 20s  TODO: Adjust value
MAX_CHASE_OFFSET = CHASE_DURATION / 10 # Maximum value that can be added to the chase duration
SCATTER_DURATION = 70*8 # Chosen randomly for now, should be 7s  TODO: Adjust value
MAX_SCATTER_OFFSET = SCATTER_DURATION / 10 # Maximum value that can be added to the scatter duration
FRIGHTENED_DURATION = 62*8 # Duration of power pellet effect in frames (x8 steps)
BLINKING_DURATION = 10*8
ENJAILED_DURATION = 120 # in steps

# FRUITS
class FruitType(IntEnum):
    CHERRY = 0
    STRAWBERRY = 1
    ORANGE = 2
    PRETZEL = 3
    APPLE = 4
    PEAR = 5
    BANANA = 6
    NONE = 7
FRUIT_SCORES = jnp.array((100, 200, 500, 700, 1000, 2000, 5000))
FRUIT_SPAWN_THRESHOLDS = jnp.array([70, 170])
FRUIT_ESCAPE_TIMEOUT = 10*8 # Chosen randomly for now, should follow a hardcoded path instead

# POSITIONS
PPX0 = 8
PPX1 = 148
PPY0 = 20
PPY1 = 152
POWER_PELLET_POSITIONS = [[PPX0, PPY0], [PPX1, PPY0], [PPX0, PPY1], [PPX1, PPY1]]
INITIAL_GHOSTS_POSITIONS = jnp.array([[75, 54], [50, 78], [40, 78], [120, 78]])
INITIAL_PACMAN_POSITION = jnp.array([75, 102])
SCATTER_TARGETS = jnp.array([
    [MsPacmanMaze.MAZE_WIDTH - 1, 0],                               # Upper right corner - Blinky
    [0, 0],                                                         # Upper left corner - Pinky
    [MsPacmanMaze.MAZE_WIDTH - 1, MsPacmanMaze.MAZE_HEIGHT - 1],    # Lower right corner - Inky
    [0, MsPacmanMaze.MAZE_HEIGHT - 1]                               # Lower left corner - Sue
])

# DIRECTIONS
DIRECTIONS = jnp.array([
    (0, 0),   # NOOP
    (0, 0),   # FIRE
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (-1, 0),  # LEFT
    (0, 1)    # DOWN
])
DIR_UP = 2
DIR_RIGHT = 3
DIR_LEFT = 4
DIR_DOWN = 5
INV_DIR = {
    2:5,
    3:4,
    4:3,
    5:2
}
INITIAL_PACMAN_DIRECTION = DIRECTIONS[DIR_LEFT]
INITIAL_LAST_DIR_INT = jnp.array(2) # LEFT
INITIAL_ACTION = jnp.array(4) # LEFT

# POINTS
PELLET_POINTS = 10
POWER_PELLET_POINTS = 50
FRUITS_POINTS = jnp.array([100, 200, 500, 700, 1000, 2000, 5000]) # cherry, strawberry, orange, pretzel, apple, pear, banana
EAT_GHOSTS_BASE_POINTS = 200

# COLORS
PATH_COLOR = jnp.array([0, 28, 136], dtype=jnp.uint8)
WALL_COLOR = jnp.array([228, 111, 111], dtype=jnp.uint8)
PELLET_COLOR = WALL_COLOR  # Same color as walls for pellets
POWER_PELLET_SPRITE = jnp.tile(jnp.concatenate([PELLET_COLOR, jnp.array([255], dtype=jnp.uint8)]), (4, 7, 1))  # 4x7 sprite
PACMAN_COLOR = jnp.array([210, 164, 74, 255], dtype=jnp.uint8)
TRANSPARENT = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)


# --- CLASSES --- #
class LevelState(NamedTuple):
    maze_layout: chex.Array # Whether the level is completed
    dofmaze: chex.Array # Precomputed degree of freedom maze layout
    pellets: chex.Array  # 2D grid of 0 (empty) or 1 (pellet)
    collected_pellets: chex.Array  # the number of pellets collected
    power_pellets: chex.Array
    current_fruit: chex.Array # The reward for the current fruit - if 0 no fruit is present
    completed_level: chex.Array  # Whether the level is completed

class GhostState(NamedTuple):
    type: GhostType
    position: chex.Array  # (x, y)
    direction: chex.Array  # (dx, dy)
    mode: chex.Array
    timer: chex.Array

class PlayerState(NamedTuple):
    position: chex.Array  # (x, y)
    direction: chex.Array  # (dx, dy)
    last_dir_int : chex.Array  # Last direction as an integer (0: UP, 1: RIGHT, 2: LEFT, 3: DOWN)
    has_pellet: chex.Array  # Boolean indicating if pacman just collected a pellet
    eaten_ghosts: chex.Array  # timers indicating which ghosts have been eaten, when set to one, does not go down, indicate respawned ghost
    power_mode_timer: chex.Array # Timer for power mode, decrements every 8 steps
    death_timer: chex.Array  # Frames left in death animation

class PacmanState(NamedTuple):
    level: LevelState
    player: PlayerState
    ghosts: Tuple[GhostState, GhostState, GhostState, GhostState] # 4 ghosts
    current_action: chex.Array # 0: NOOP, 1: NOOP, 2: UP ...
    step_count: chex.Array
    level_num: chex.Array
    lives: chex.Array  # Number of lives left
    score: chex.Array
    score_changed: chex.Array
    game_over: chex.Array
    reset: chex.Array  # Reset state for the next episode

class PacmanObservation(NamedTuple):
    grid: chex.Array  # 2D array showing layout of walls, pellets, pacman, ghosts

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array


class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo]):
    def __init__(self):
        super().__init__()
        self.skipped = False
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
        state = PacmanState(
            level = LevelState(
                maze_layout=RESET_MAZE,
                dofmaze=MsPacmanMaze.precompute_dof(MsPacmanMaze.MAZES[RESET_MAZE]), # Precompute degree of freedom maze layout
                pellets=jnp.copy(MsPacmanMaze.BASE_PELLETS),
                collected_pellets=jnp.array(0).astype(jnp.uint8),
                power_pellets=jnp.ones(4, dtype=jnp.bool_),
                current_fruit=jnp.array(FruitType.NONE).astype(jnp.uint8),
                completed_level=jnp.array(False, dtype=jnp.bool_)
            ),
            player = PlayerState(
                position=INITIAL_PACMAN_POSITION,
                direction=INITIAL_PACMAN_DIRECTION,
                last_dir_int=INITIAL_LAST_DIR_INT,
                has_pellet=jnp.array(False),
                eaten_ghosts=jnp.array(0).astype(jnp.uint8), # number of eaten ghost since power pellet consumed
                power_mode_timer=jnp.array(0).astype(jnp.uint8),  # Timer for power mode,
                death_timer=jnp.array(0),
            ),
            ghosts = tuple(
                GhostState(
                    type=GhostType(i),
                    position=INITIAL_GHOSTS_POSITIONS[i],
                    direction=jnp.zeros(2, dtype=jnp.int8),
                    mode=(jnp.array(GhostMode.RANDOM).astype(jnp.uint8) if i < 2 
                          else jnp.array(GhostMode.SCATTER).astype(jnp.uint8)),
                    timer=jnp.array(SCATTER_DURATION).astype(jnp.uint8),
                ) for i in range(4)
            ),
            current_action = INITIAL_ACTION,
            score=jnp.array(0),
            score_changed=jnp.zeros(MAX_SCORE_DIGITS, dtype=jnp.bool_), # indicates which score digit changed since the last step
            step_count=jnp.array(0),
            game_over=jnp.array(False),
            level_num=1,
            lives=jnp.array(INITIAL_LIVES, dtype=jnp.int8),  # Number of lives left
            reset=jnp.array(True, dtype=jnp.bool_) # to reload the background
        )
        obs = None
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array, key: chex.PRNGKey) -> tuple[
        PacmanObservation, PacmanState, jax.Array, jax.Array, PacmanInfo]:
        # Skip current step if in reset state, so the renderer has time to react
        if state.reset and self.skipped == False:
            self.skipped = True
            obs = None
            reward = 0.0
            done = jnp.array(False, dtype=jnp.bool_)
            info = PacmanInfo(score=state.score, done=done)
            return obs, state, reward, done, info
        self.skipped = False
        
        pacman_pos = state.player.position
        pacman_dir = state.player.direction
        pacman_last_dir_int = state.player.last_dir_int
        power_mode_timer = state.player.power_mode_timer

        ghost_positions = jnp.array([ghost.position for ghost in state.ghosts])
        ghosts_dirs = jnp.array([ghost.direction for ghost in state.ghosts])
        dofmaze = state.level.dofmaze
        completed_level = jnp.array(False, dtype=jnp.bool_)
        game_over = jnp.array(False, dtype=jnp.bool_)

        # If in death animation, decrement timer and freeze everything
        if state.player.death_timer > 0:
            new_death_timer = state.player.death_timer - 1
            # When timer reaches 0, reset positions if lives remain or set state to game over
            if new_death_timer == 0:
                if state.lives >= 0:
                    power_mode_timer = jnp.array(0).astype(jnp.uint8)  # Reset power mode timer
                    pacman_pos = INITIAL_PACMAN_POSITION
                    pacman_dir = INITIAL_PACMAN_DIRECTION
                    ghost_positions = INITIAL_GHOSTS_POSITIONS
                    ghosts_dirs = jnp.zeros_like(ghost_positions)
                else:
                    game_over = jnp.array(True, dtype=jnp.bool_)

            new_state = PacmanState(
                level = LevelState(
                    maze_layout=state.level.maze_layout,
                    dofmaze=dofmaze,
                    pellets=state.level.pellets,
                    collected_pellets=state.level.collected_pellets,
                    power_pellets=state.level.power_pellets,
                    current_fruit=state.level.current_fruit,
                    completed_level=completed_level
                ),
                player = PlayerState(
                    position=pacman_pos,
                    direction=pacman_dir,
                    last_dir_int=jnp.array(2),
                    has_pellet=state.player.has_pellet,
                    eaten_ghosts=state.player.eaten_ghosts,
                    power_mode_timer=power_mode_timer,
                    death_timer=new_death_timer,
                ),
                ghosts = tuple(
                    GhostState(
                        type=GhostType(i),
                        position=ghost_positions[i],
                        direction=ghosts_dirs[i],
                        mode=jnp.array(0),
                        timer=jnp.array(0),
                    ) for i in range(4)
                ),
                current_action=state.current_action,
                score=state.score,
                score_changed=jnp.zeros(MAX_SCORE_DIGITS, dtype=jnp.bool_),
                step_count=state.step_count + 1,
                game_over=game_over,
                level_num=state.level_num,
                lives=state.lives,
                reset=jnp.array(False, dtype=jnp.bool_)
            )
            obs = None
            reward = state.score
            done = game_over
            info = PacmanInfo(score=state.score, done=done)
            return obs, new_state, reward, done, info

        # Pacman movement
        action = last_pressed_action(action, state.current_action)
        possible_directions = available_directions(state.player.position, state.level.dofmaze)
        if action < 0 or action > len(DIRECTIONS) - 1: # Ignore illegal actions
            action = Action.NOOP
        if action != Action.NOOP and action != Action.FIRE and possible_directions[action - 2]:
            new_pacman_dir = DIRECTIONS[action]
            executed_action = action
            pacman_last_dir_int = action - 2  # Convert action to direction index (0: UP, 1: RIGHT, 2: LEFT, 3: DOWN)
        else:
            # check for wall deadly_collision
            if state.current_action > 1 and stop_wall(state.player.position, dofmaze)[state.current_action - 2]:
                executed_action = 0
                new_pacman_dir = jnp.array([0, 0])
            else:
                executed_action = state.current_action
                new_pacman_dir = state.player.direction
        # if state.step_count % 2:
        if True:
            new_pacman_pos = state.player.position + new_pacman_dir
            new_pacman_pos = new_pacman_pos.at[0].set(new_pacman_pos[0] % 160)
        else:
            new_pacman_pos = state.pacman_pos

        # power pellets 
        collected_pellets = state.level.collected_pellets
        power_pellets = state.level.power_pellets
        current_fruit = state.level.current_fruit
        px, py = new_pacman_pos // 4
        ate_power_pill = False
        if px == 1:
            if (py == 3 or py == 4) and power_pellets[0]:
                power_pellets = state.level.power_pellets.at[0].set(False)
                ate_power_pill = True
            elif (py == 36 or py == 37) and power_pellets[2]:
                power_pellets = state.level.power_pellets.at[2].set(False)
                ate_power_pill = True
        elif px == 36:
            if (py == 3 or py == 4) and power_pellets[1]:
                power_pellets = state.level.power_pellets.at[1].set(False)
                ate_power_pill = True
            elif (py == 36 or py == 37) and power_pellets[3]:
                power_pellets = state.level.power_pellets.at[3].set(False)
                ate_power_pill = True
        if ate_power_pill:
            collected_pellets += 1
        elif state.player.power_mode_timer > 0 and state.step_count % 8 == 0:
            # Decrement power mode timer
            power_mode_timer = state.player.power_mode_timer - 1
        pellets = state.level.pellets
        # Check for pellet consumption
        has_pellet = jnp.array(False)
        x_offset = 5 if new_pacman_pos[0] < 75 else 1
        if new_pacman_pos[0] % 8 == x_offset and new_pacman_pos[1] % 12 == 6:
            x_pellets = (new_pacman_pos[0] - 2) // 8
            y_pellets = (new_pacman_pos[1] + 4) // 12
            if state.level.pellets[x_pellets, y_pellets]:
                has_pellet = jnp.array(True)
                pellets = state.level.pellets.at[x_pellets, y_pellets].set(False)
        score = state.score + jax.lax.select(has_pellet, PELLET_POINTS, 0)
        if has_pellet:
            print(f"Pacman collected a pellet at {new_pacman_pos}, score: {score}")
            collected_pellets += 1

        # TODO: Optionally implement 'previous_fruit' with special orange/banana only once per level (only cited by one source)
        # TODO: Implement fruit consumption 
        for threshold in FRUIT_SPAWN_THRESHOLDS:
            if collected_pellets == threshold:
                current_fruit = get_level_fruit(state.level, key)

        ghost_positions, ghosts_dirs, ghosts_modes, ghosts_timers = ghosts_step(
            state.ghosts, state.player, ate_power_pill, dofmaze, key
        )
        # Collision detection
        eaten_ghosts = state.player.eaten_ghosts
        deadly_collision = False
        for i in range(4): 
            if jnp.all(abs(new_pacman_pos - ghost_positions[i]) < 8):
                if ghosts_modes[i] == GhostMode.FRIGHTENED or ghosts_modes[i] == GhostMode.BLINKING:  # If are frighted
                    # Ghost eaten
                    score += EAT_GHOSTS_BASE_POINTS * (2 ** eaten_ghosts)   # TODO: Reset eaten_ghosts after power mode ends
                    ghost_positions = ghost_positions.at[i].set((76, 70))  # Reset eaten ghost position
                    ghosts_dirs = ghosts_dirs.at[i].set([0, 0])  # Reset eaten ghost direction
                    ghosts_modes = ghosts_modes.at[i].set(GhostMode.ENJAILED.value)
                    ghosts_timers = ghosts_timers.at[i].set(ENJAILED_DURATION)
                    eaten_ghosts = eaten_ghosts + 1
                else:
                    deadly_collision = True  # Collision with an already eaten and respawned ghost
                    
        else:
            deadly_collision = jnp.any(jnp.all(abs(new_pacman_pos - ghost_positions) < 8, axis=1))
        new_lives = state.lives - jnp.where(deadly_collision, 1, 0)
        new_death_timer = jnp.where(deadly_collision, RESET_TIMER, 0)
        maze_layout = state.level.maze_layout
        if collected_pellets >= PELLETS_TO_COLLECT: # TODO: Fix! Is never true.
            # If all pellets collected, reset game
            collected_pellets = jnp.array(0, dtype=jnp.uint8)
            ghost_positions = INITIAL_GHOSTS_POSITIONS
            ghosts_dirs = jnp.zeros_like(ghost_positions)
            power_mode_timer = jnp.array(0, dtype=jnp.uint8)
            power_pellets = jnp.ones(4, dtype=jnp.bool_)
            completed_level = jnp.array(True)
            score += 500
            pellets = jnp.copy(MsPacmanMaze.BASE_PELLETS)  # Reset pellets
            maze_layout = (maze_layout + 1) % 4  # len(MAZES)
            print(f"Level completed! New level: {maze_layout}")
            dofmaze= MsPacmanMaze.precompute_dof(MsPacmanMaze.MAZES[maze_layout])

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
            level = LevelState(
                maze_layout=maze_layout,
                dofmaze=dofmaze,
                pellets=pellets,
                collected_pellets=collected_pellets,
                power_pellets=power_pellets,
                current_fruit=current_fruit,
                completed_level=completed_level
            ),
            player = PlayerState(
                position=new_pacman_pos,
                direction=new_pacman_dir,
                last_dir_int=pacman_last_dir_int,
                has_pellet=has_pellet,
                eaten_ghosts=eaten_ghosts,
                power_mode_timer=power_mode_timer,
                death_timer=new_death_timer,
            ),
            ghosts = tuple(
                GhostState(
                    type=GhostType(i),
                    position=ghost_positions[i],
                    direction=ghosts_dirs[i],
                    mode=ghosts_modes[i],
                    timer=ghosts_timers[i],
                ) for i in range(4)
            ),
            current_action=executed_action,
            score=score,
            score_changed=score_changed,
            step_count=state.step_count + 1,
            game_over=game_over,
            level_num=state.level_num, # TODO: Increment level_num
            lives=new_lives,
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
        self.sprites = MsPacmanRenderer.load_sprites()

    def reset_bg(self):
        """Reset the background for a new level."""
        self.SPRITE_BG = MsPacmanMaze.load_background(RESET_MAZE)
        self.SPRITE_BG = MsPacmanRenderer.render_score(self.SPRITE_BG, 0, jnp.arange(MAX_SCORE_DIGITS) == (MAX_SCORE_DIGITS - 1), self.sprites["score"])
        self.SPRITE_BG = MsPacmanRenderer.render_lives(self.SPRITE_BG, INITIAL_LIVES, self.sprites["pacman"][1][1]) # Life sprite (right looking pacman)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PacmanState):
        if state.reset:
            # Render game over screen
            self.reset_bg()
        if state.level.completed_level:
            self.SPRITE_BG = MsPacmanMaze.load_background(state.level.maze_layout)
        raster = self.SPRITE_BG
        # de-render pellets
        # if state.has_pellet:
        if state.player.has_pellet:
            pellet_x = state.player.position[0] + 3
            pellet_y = state.player.position[1] + 4
            for i in range(4):
                for j in range(2):
                    self.SPRITE_BG = self.SPRITE_BG.at[pellet_x+i, pellet_y+j].set(PATH_COLOR)
        # power pellets
        for i in range(2):
            pel_n = 2*i + ((state.step_count & 0b1000) >> 3) # Alternate power pellet rendering
            if state.level.power_pellets[pel_n]:
                pellet_x, pellet_y = POWER_PELLET_POSITIONS[pel_n]
                raster = aj.render_at(raster, pellet_x, pellet_y, POWER_PELLET_SPRITE)
        # Render pacman
        orientation = state.player.last_dir_int
        pacman_sprite = self.sprites["pacman"][orientation][((state.step_count & 0b1000) >> 2)]
        raster = aj.render_at(raster, state.player.position[0], state.player.position[1], 
                              pacman_sprite)
        ghosts_orientation = ((state.step_count & 0b10000) >> 4) # (state.step_count % 32) // 16

        # Render score if changed
        if jnp.any(state.score_changed):
            self.SPRITE_BG = MsPacmanRenderer.render_score(self.SPRITE_BG, state.score, state.score_changed, self.sprites["score"])

        for i, ghost in enumerate(state.ghosts):
            # Render frightened ghost
            if not (state.ghosts[i].mode == GhostMode.FRIGHTENED or state.ghosts[i].mode == GhostMode.BLINKING):
                g_sprite = self.sprites["ghost"][ghosts_orientation][i]
            elif state.ghosts[i].mode == GhostMode.BLINKING and ((state.step_count & 0b1000) >> 3):
                g_sprite = self.sprites["ghost"][ghosts_orientation][5] # white blinking effect
            else:
                g_sprite = self.sprites["ghost"][ghosts_orientation][4] # blue ghost
            raster = aj.render_at(raster, ghost.position[0], ghost.position[1], g_sprite)

        # Remove one life if a life is lost
        if state.player.death_timer == RESET_TIMER-1:
            self.SPRITE_BG = MsPacmanRenderer.render_lives(self.SPRITE_BG, state.lives, self.sprites["pacman"][1][1])
        return raster
    
    @staticmethod
    def render_score(raster, score, score_changed, digit_sprites, score_x=60, score_y=190, spacing=1, bg_color=jnp.array([0, 0, 0])):
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

    @staticmethod
    def render_lives(raster, current_lives, life_sprite, initial_lives=INITIAL_LIVES, life_x=12, life_y=182, spacing=4, bg_color=jnp.array([0, 0, 0])):
        """
        Render the lives on the raster at a fixed position.
        """
        if current_lives == initial_lives:
            for i in range(current_lives):
                raster = aj.render_at(raster, life_x + i * (life_sprite.shape[1] + spacing), life_y, life_sprite)
        elif current_lives >= 0:
            bg_sprite = jnp.full(life_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
            raster = aj.render_at(raster, life_x + current_lives * (life_sprite.shape[1] + spacing), life_y, bg_sprite)
        return raster
    
    # TODO: Implement!
    @staticmethod
    def render_fruit(raster, level, fruit_timer,  fruit_sprites):
        ...
    
    @staticmethod
    def load_sprites() -> dict[str, Any]:
        SPRITE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/mspacman"
        sprites: Dict[str, Any] = {}
     
        # Helper function to load a single sprite frame
        def load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(SPRITE_PATH, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)
            return None
    
        # List of alls sprite names
        sprite_names = [ # The order here is important as it determines the sprites index
            'fruit_cherry','fruit_strawberry','fruit_orange',
            'fruit_pretzel','fruit_apple','fruit_pear','fruit_banana',
            'ghost_blinky','ghost_pinky','ghost_inky','ghost_sue','ghost_blue','ghost_white',
            'pacman_0','pacman_1','pacman_2','pacman_3',
            'score_0','score_1','score_2','score_3','score_4',
            'score_5','score_6','score_7','score_8','score_9'
        ]
        
        # Load raw sprites
        fruits = []
        ghosts = []
        pacmans = []
        score = []
        for name in sprite_names:
            loaded_sprite = load_sprite_frame(name)
            if loaded_sprite is not None:
                if "fruit" in name:
                    fruits.append(loaded_sprite)
                if "ghost" in name:
                    ghosts.append(loaded_sprite)
                elif "pacman" in name:
                    pacmans.append(loaded_sprite)
                elif "score" in name:
                    score.append(loaded_sprite)

        # Postprocess fruit sprites
        padded_fruits = aj.pad_to_match(fruits)
        jax_fruits = jnp.stack(padded_fruits)
        # Postprocess ghost sprites
        symmetric_ghosts = [jnp.flipud(ghost) for ghost in ghosts]
        jax_ghosts = [jnp.array(ghosts), jnp.array(symmetric_ghosts)]
        # Postprocess pacman sprites
        pacmans_right = [jnp.flipud(p) for p in pacmans]
        pacmans_up = [jnp.rot90(p, 3) for p in pacmans_right]
        pacmans_down = [jnp.rot90(p) for p in pacmans_right]
        jax_pacmans = [pacmans_up, pacmans_right, pacmans, pacmans_down]
        # Postprocess score sprites
        padded_score = aj.pad_to_match(score)
        jax_score = jnp.stack(padded_score)

        # Save resulting sprites
        sprites["fruit"] = jax_fruits
        sprites["ghost"] = jax_ghosts
        sprites["pacman"] = jax_pacmans
        sprites["score"] = jax_score
        return sprites


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


def get_allowed_directions(position: chex.Array, direction: chex.Array, dofmaze: chex.Array):
    """
    Returns an array of all directions (indices) in which movement is possible.
    Turning is only allowed at the centre of each tile and reverting is not allowed.
    """
    allowed = []
    direction_idx = get_direction_index(direction)
    if position[0] % 4 == 1 or position[1] % 12 == 6: # on horizontal or vertical grid - tile centre
        direction_indices = [DIR_UP, DIR_RIGHT, DIR_LEFT, DIR_DOWN]
        possible = available_directions(position, dofmaze) 
        for i, can_go in zip(direction_indices, possible):
            if can_go and (direction_idx == 0 or i != INV_DIR.get(direction_idx, -1)):
                allowed.append(i)
    else:
        allowed.append(direction_idx)
    return allowed


def get_chase_target(ghost: GhostType,
                     ghost_position: chex.Array, blinky_pos: chex.Array,
                     player_pos: chex.Array, player_dir: chex.Array) -> chex.Array:
    """
    Compute the chase-mode target for each ghost:
    0=Red (Blinky), 1=Pink (Pinky), 2=Blue (Inky), 3=Orange (Sue)
    """
    match ghost:
        case GhostType.BLINKY:
            # Target Pac-Man's current tile
            return player_pos
        case GhostType.PINKY:
            # Target 4 tiles ahead of Pac-Man
            ahead = player_pos + 4 * player_dir
            return ahead
        case GhostType.INKY:
            # Target the tip of the vector from Blinky to two tiles ahead of Pac-Man, doubled
            two_ahead = player_pos + 2 * player_dir
            vect = two_ahead - blinky_pos
            return blinky_pos + 2 * vect
        case GhostType.SUE:
            # Target Pac-Man if >8 tiles away, else target corner
            dist = jnp.linalg.norm(ghost_position - player_pos)
            return jnp.where(dist > 8, player_pos, SCATTER_TARGETS[GhostType.SUE])


def ghosts_step(ghosts: GhostState[4], player: PlayerState, ate_power_pill: chex.Array, dofmaze: chex.Array, key: chex.Array
                ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Step all ghosts. key can be a PRNGKey or None for deterministic.
    """
    n_ghosts = len(ghosts)
    keys = jax.random.split(key, n_ghosts)
    new_positions = []
    new_dirs = []
    modes = []
    timers = []
    chase_offset = jax.random.randint(keys[0], (), 0, MAX_CHASE_OFFSET)
    scatter_offset = jax.random.randint(keys[0], (), 0, MAX_SCATTER_OFFSET)
    for i in range(n_ghosts):
        chase_target = get_chase_target(ghosts[i].type, ghosts[i].position, ghosts[GhostType.BLINKY].position,
                                        player.position, player.direction)
        pos, dir, mode, timer = ghost_step(ghosts[i], ate_power_pill, dofmaze, keys[i],
                                           chase_target, chase_offset, scatter_offset)
        new_positions.append(pos)
        new_dirs.append(dir)
        modes.append(mode)
        timers.append(timer)
    return jnp.stack(new_positions), jnp.stack(new_dirs), jnp.array(modes), jnp.array(timers)


def ghost_step(ghost: GhostState, ate_power_pill: chex.Array, dofmaze: chex.Array, key: chex.Array,
               chase_target: chex.Array, chase_offset: chex.Array, scatter_offset: chex.Array
               ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Step function for a single ghost. Never stops, never reverses, can change direction at intersections.
    """
    """
    GHOST BEHAVIOUR
    -   All Ghosts always aim for a specific target
    -   They try to minimize their horizontal and vertical distance to this target
    -   They prioritizes the longer one of those two distances - If equal, they choose randomly
    -   They can only change direction when they are on a vertical or horizontal grid
    -   They cannot reverse direction without a mode change

    MODES
    -   CHASE: Aim for their specific target
    -   SCATTER: Aim for their specific corner target (BLINKY and PINKY move randomly for the first scatter of each game)   (EXAMINE FURTHER)
    -   FRIGHTENED: Move randomly
    -   Every mode change from CHASE to SCATTER, SCATTER to CHASE and any mode to FRIGHTENED causes a reversal of direction
    -   Mode changes happen on a timer with some additional randomness

    BLINKY (Red)
    -   Aims for Ms Pacman directly

    PINKY (Pink)
    -   Aims for a spot 4 tiles in front of Ms Pacman
    -   If Ms Pacman is looking up it aims for a spot 4 tiles up and 4 tiles left of Ms Pacman
        (due to a bug in the original game, not implemented here)

    INKY (Teal)
    -   Aims for a spot determined by Blinkys position relative to Ms Pacman
    -   Draw a straight line from Blinkys position to the spot 2 tiles in front of Ms Pacman
        and extend it equally as far onwards to determine the aiming point

    SUE (Orange)
    -   Aims for Ms Pacman directly if further away than 8 tiles (euclidian distance)
    -   When closer than 8 tiles it aims for the lower left corner

    SOURCES:
    https://www.classicarcadegaming.com/forums/index.php?topic=6701.0
    https://en.wikipedia.org/wiki/Ms._Pac-Man?utm_source=chatgpt.com
    https://www.youtube.com/watch?v=ICwzQ0_RCcQ&t
    https://www.youtube.com/watch?v=sQK7PmR8kpQ
    https://www.youtube.com/watch?v=VV4_kVIV9WE

    ADDITIONAL RESOURCES:
    https://www.gamedeveloper.com/design/the-pac-man-dossier
    https://www.researchgate.net/publication/224180057_Ghost_Direction_Detection_and_other_Innovations_for_Ms_Pac-Man
    https://nn.cs.utexas.edu/downloads/papers/schrum.tciaig16.pdf?utm_source=chatgpt.com
    http://donhodges.com/pacman_pinky_explanation.htm
    http://donhodges.com/ms_pacman_bugs.htm
    http://donhodges.com/how_high_can_you_get3.htm
    http://cubeman.org/arcade-source/mspac.asm
    """

    # 0) Initialize
    new_pos = ghost.position
    new_dir = ghost.direction
    new_mode = ghost.mode
    last_timer = ghost.timer
    new_timer = ghost.timer
    if ghost.timer > 0:
        new_timer = ghost.timer - 1
    revert = False

    # 1) Update ghost mode and timer
    if ate_power_pill and ghost.mode not in [GhostMode.ENJAILED, GhostMode.RETURNING]:
        new_mode = GhostMode.FRIGHTENED
        new_timer = FRIGHTENED_DURATION
        revert = True
    elif new_timer == 0 and last_timer > 0:
        match ghost.mode:
            case GhostMode.CHASE:
                new_mode = GhostMode.SCATTER
                new_timer = SCATTER_DURATION + chase_offset
                revert = True
            case GhostMode.SCATTER:
                new_mode = GhostMode.CHASE
                new_timer = CHASE_DURATION + scatter_offset
                revert = True
            case GhostMode.FRIGHTENED:
                new_mode = GhostMode.BLINKING
                new_timer = BLINKING_DURATION
            case GhostMode.RETURNING:
                new_mode = GhostMode.ENJAILED
                new_timer = ENJAILED_DURATION
            case GhostMode.BLINKING | GhostMode.ENJAILED | GhostMode.RANDOM:
                new_mode = GhostMode.CHASE
                new_timer = CHASE_DURATION

    # 2) Update ghost direction
    if revert:
        new_dir = DIRECTIONS[INV_DIR[get_direction_index(ghost.direction)]]
    elif new_mode == GhostMode.ENJAILED | new_mode == GhostMode.RETURNING:
        pass
    else:
        # 2.2) Choose new direction
        allowed = get_allowed_directions(ghost.position, ghost.direction, dofmaze)
        if not allowed: # If no direction is allowed - continue forward
            pass
        elif len(allowed) == 1: # If only one allowed direction - take it
            new_dir = DIRECTIONS[allowed[0]]
        else:
            match new_mode:
                case GhostMode.CHASE:
                    directions = get_target_direction_indices(ghost.position, chase_target)
                    new_dir = select_target_direction(directions, allowed, key)
                case GhostMode.SCATTER:
                    target = SCATTER_TARGETS[ghost.type]
                    directions = get_target_direction_indices(ghost.position, target)
                    new_dir = select_target_direction(directions, allowed, key)
                case GhostMode.FRIGHTENED | GhostMode.BLINKING | GhostMode.RANDOM:
                    new_dir = DIRECTIONS[jax.random.choice(key, jnp.array(allowed))]

    # 3) Update ghost position
    new_pos = ghost.position + new_dir
    new_pos = new_pos.at[0].set(new_pos[0] % 160) # wrap horizontally
    return new_pos, new_dir, new_mode, new_timer


def get_target_direction_indices(position: chex.Array, target: chex.Array) -> list:
    """
    Returns the directions (indices) which should be taken to minimize the horizontal or vertical distance to the target.
    Prioritizes the bigger distance and returns both possible directions they are equal.
    """
    horizontal_distance = jnp.abs(position[0] - target[0])
    vertical_distance = jnp.abs(position[1] - target[1])
    directions = []

    if horizontal_distance >= vertical_distance:
        if position[0] < target[0]:
            directions.append(DIR_RIGHT)
        else:
            directions.append(DIR_LEFT)
    elif vertical_distance >= horizontal_distance:
        if position[1] < target[1]:
            directions.append(DIR_DOWN)
        else:
            directions.append(DIR_UP)
    return directions


def select_target_direction(directions: chex.Array, allowed: chex.Array, key: chex.Array):
    """
    Returns the direction (tuple) that should be taken by a ghost, given the preferred directions towards its target
    and the allowed directions at its current position.
    """
    possible = []
    for direction in directions:
        if jnp.any(allowed == direction):
            possible.append(direction)

    if len(possible) == 0:
        if len(directions) == 0:
            raise ValueError("No path found!")
        else:
            filtered = [d for d in allowed if d != INV_DIR[directions[0]]]
            return DIRECTIONS[jax.random.choice(key, jnp.array(filtered))]
    elif len(possible) == 1:
        return DIRECTIONS[possible[0]]
    else:
        return DIRECTIONS[jax.random.choice(key, jnp.array(possible))]


def get_level_maze(level: chex.Array):
    if level < 0:
        raise ValueError("Invalid level!")
    elif level < 3:
        return 0
    elif level < 6:
        return 1
    elif level < 10:
        return 2
    elif level < 14:
        return 3
    else:
        if level % 4 == 0 or level % 4 == 1:
            return 2
        else:
            return 3


def get_level_fruit(level: chex, key: chex.Array):
    match level:
        case 1: return FruitType.CHERRY
        case 2: return FruitType.STRAWBERRY
        case 3: return FruitType.ORANGE
        case 4: return FruitType.PRETZEL
        case 5: return FruitType.APPLE
        case 6: return FruitType.PEAR
        case 7: return FruitType.BANANA
        case _: return jax.random.randint(key, (), 1, 8)


def get_fruit_spawn(level: chex.Array, key: chex.Array):
    fruit = get_level_fruit(level, key)
    maze = get_level_maze(level)
    tunnel_heights = MsPacmanMaze.TUNNEL_HEIGHTS[maze]
    tunnel_positions = [[0, tunnel_heights[0]], [MsPacmanMaze.MAZE_WIDTH - 1, tunnel_heights[0]],
                        [0, tunnel_heights[1]], [MsPacmanMaze.MAZE_WIDTH - 1, tunnel_heights[1]]]
    if tunnel_heights[1] == 0:
        tunnel_idx = jax.random.randint(key, (), 0, 2)
    else:
        tunnel_idx = jax.random.randint(key, (), 0, 4)
    spawn_point = tunnel_positions[tunnel_idx]
    return spawn_point, fruit
