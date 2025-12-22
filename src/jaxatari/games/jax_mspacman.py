"""
Group: Sooraj Rathore, Kadir Özen
Edit: Jan Rafflewski
"""

"""
TODO
    1)  [x] Validate ghost behaviour
    2)  [x] Level progression
    3)  [x] Performance improvements
    4)  [ ] JIT compatibility

    Optional:
    a)  [ ] Correct speed
    b)  [ ] Pacman death animation
    c)  [ ] Bonus Life at 10000 score
    d)  [ ] Fruit scale and animation
    e)  [ ] Fruit movement patterns
    f)  [ ] Ghost starting positions
    g)  [ ] Ghost behavioral quirks
    h)  [ ] Ghost enjailment
    i)  [ ] Ghost timings
    j)  [ ] Correct maze colors
"""


# -------- Imports --------
from enum import IntEnum
import os
from functools import partial
import time
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.games.mspacman_mazes import MsPacmanMaze


# -------- Enums -------- 
class FruitType(IntEnum):
    CHERRY = 0
    STRAWBERRY = 1
    ORANGE = 2
    PRETZEL = 3
    APPLE = 4
    PEAR = 5
    BANANA = 6
    NONE = 7

class GhostType(IntEnum):
    BLINKY = 0
    PINKY = 1
    INKY = 2
    SUE = 3

class GhostMode(IntEnum):
    RANDOM = 0
    CHASE = 1
    SCATTER = 2
    FRIGHTENED = 3
    BLINKING = 4
    RETURNING = 5
    ENJAILED = 6


# -------- Constants --------
# GENERAL
RESET_LEVEL = 1 # the starting level, loaded when reset is called
RESET_TIMER = 40  # Timer for resetting the game after death
MAX_SCORE_DIGITS = 6 # Number of digits to display in the score
MAX_LIVE_COUNT = 8
PELLETS_TO_COLLECT = 154  # Total pellets to collect in the maze (including power pellets)
# PELLETS_TO_COLLECT = 4
INITIAL_LIVES = 2 # Number of starting bonus lives
BONUS_LIFE_LIMIT = 10000 # Maximum number of bonus lives
COLLISION_THRESHOLD = 8 # Contacts below this distance count as collision

# GHOST TIMINGS
CHASE_DURATION = 20*4*8 # Estimated for now, should be 20s  TODO: Adjust value
MAX_CHASE_OFFSET = CHASE_DURATION / 10 # Maximum value that can be added to the chase duration
SCATTER_DURATION = 7*4*8 # Estimated for now, should be 7s  TODO: Adjust value
# SCATTER_DURATION = 50
MAX_SCATTER_OFFSET = SCATTER_DURATION / 10 # Maximum value that can be added to the scatter duration
FRIGHTENED_DURATION = 62*8 # Duration of power pellet effect in frames (x8 steps)
BLINKING_DURATION = 10*8
ENJAILED_DURATION = 120 # in steps
RETURN_DURATION = 2*8 # Estimated for now, should be as long as it takes the ghost to return from jail to the path TODO: Adjust value

# FRUITS
FRUIT_SPAWN_THRESHOLDS = jnp.array([50, 100]) # The original was more like ~70, ~170 but this version has a reduced number of pellets
# FRUIT_SPAWN_THRESHOLDS = jnp.array([4, 100])
FRUIT_WANDER_DURATION = 20*8 # Chosen randomly for now, should follow a hardcoded path instead

# POSITIONS
POWER_PELLET_TILES = jnp.array([[1, 3], [36, 3], [1, 36], [36, 36]])
POWER_PELLET_HITBOXES = jnp.array([[1, 3], [36, 3], [1, 36], [36, 36],
                                   [1, 4], [36, 4], [1, 37], [36, 37]])
INITIAL_GHOSTS_POSITIONS = jnp.array([[73, 54], [49, 78], [41, 78], [121, 78]])
INITIAL_PACMAN_POSITION = jnp.array([75, 102])
JAIL_POSITION = jnp.array([77, 70])
SCATTER_TARGETS = jnp.array([
    [MsPacmanMaze.WIDTH - 1, 0],                        # Upper right corner - Blinky
    [0, 0],                                             # Upper left corner - Pinky
    [MsPacmanMaze.WIDTH - 1, MsPacmanMaze.HEIGHT - 1],  # Lower right corner - Inky
    [0, MsPacmanMaze.HEIGHT - 1]                        # Lower left corner - Sue
])

# ACTIONS
DIRECTIONS = jnp.array([Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN])
ACTIONS = jnp.array([ # Translates generic JAXAtari actions to MsPacman actions
    (0, 0),   # NOOP
    (0, 0),   # FIRE (unused)
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (-1, 0),  # LEFT
    (0, 1)    # DOWN
])
INITIAL_ACTION = Action.LEFT # LEFT
INITIAL_LAST_ACTION = Action.LEFT # LEFT

# POINTS
PELLET_POINTS = 10
POWER_PELLET_POINTS = 50
FRUIT_REWARDS = jnp.array([100, 200, 500, 700, 1000, 2000, 5000]) # cherry, strawberry, orange, pretzel, apple, pear, banana
EAT_GHOSTS_BASE_POINTS = 200
LEVEL_COMPLETED_POINTS = 500

# COLORS
PATH_COLOR = jnp.array([0, 28, 136], dtype=jnp.uint8)
WALL_COLOR = jnp.array([228, 111, 111], dtype=jnp.uint8)
PELLET_COLOR = WALL_COLOR  # Same color as walls for pellets
POWER_PELLET_SPRITE = jnp.tile(jnp.concatenate([PELLET_COLOR, jnp.array([255], dtype=jnp.uint8)]), (4, 7, 1))  # 4x7 sprite
PACMAN_COLOR = jnp.array([210, 164, 74, 255], dtype=jnp.uint8)
TRANSPARENT = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)


# -------- Entity classes --------
class LevelState(NamedTuple):
    id: chex.Array                  # Int - Number of the current level, starts at 1
    collected_pellets: chex.Array   # Int - Number of collected pellets
    dofmaze: chex.Array             # Bool[x][y][4] - Precomputed degree of freedom maze layout
    pellets: chex.Array             # Bool[x][y] - 2D grid of 0 (empty) or 1 (pellet)
    power_pellets: chex.Array       # Bool[4] - Indicates wheter the power pellet is available
    loaded: chex.Array              # Int - 0: Not loaded, 1: loading, 2: loaded

class GhostsState(NamedTuple):
    positions: chex.Array           # Tuple - (x, y)
    types: chex.Array               # Enum - 0: BLINKY, 1: PINKY, 2: INKY, 3: SUE
    actions: chex.Array             # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    modes: chex.Array               # Enum - 0: RANDOM, 1: CHASE, 2: SCTATTER, 3: FRIGHTENED, 4: BLINKING, 5: RETURNING, 6: ENJAILED
    timers: chex.Array              # Int - Triggers mode change when reaching 0, decrements every step
    keys: chex.Array                # chex.PRNGKey - Unique random key, saved in state to prevent repeated generation

class PlayerState(NamedTuple):
    position: chex.Array            # Tuple - (x, y)
    action: chex.Array              # Enum - 0: NOOP, 1: FURE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    has_pellet: chex.Array          # Bool - Indicates if pacman just collected a pellet
    eaten_ghosts: chex.Array        # Int - Indicates the number of ghosts eaten since the last power pellet

class FruitState(NamedTuple):
    position: chex.Array            # Tuple - (x, y)
    exit: chex.Array                # Tuple - (x, y) Position of the tunnel through which it will exit
    type: chex.Array                # Enum - 0: CHERRY, 1: STRAWBERRY, 2: ORANGE, 3: PRETZEL, 4: APPLE, 5: PEAR, 6: BANANA, 7: NONE
    action: chex.Array              # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    timer: chex.Array               # Int - Time until leaving through the exit tunnel, decrements every step

class PacmanState(NamedTuple):
    level: LevelState               # LevelState
    player: PlayerState             # PlayerState
    ghosts: GhostsState             # GhostStates
    fruit: FruitState               # FruitState
    lives: chex.Array               # Int - Number of lives left
    score: chex.Array               # Int - Total score reached
    score_changed: chex.Array       # Bool[] - Indicates which score digit changed since the last step
    freeze_timer: chex.Array        # Int - Time until game is unfrozen, decrements every step
    step_count: chex.Array          # Int - Number of steps made in the current level

class PacmanObservation(NamedTuple):
    player_position: chex.Array
    player_action: chex.Array
    ghost_positions: chex.Array
    ghost_actions: chex.Array
    ghost_modes: chex.Array
    ghosts_eaten: chex.Array
    fruit_position: chex.Array
    fruit_action: chex.Array
    fruit_type: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array

class PacmanInfo(NamedTuple):
    level: chex.Array
    score: chex.Array
    lives: chex.Array


# -------- Game class --------
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
        """
        Resets the game to its initial state.
        """
        return None, reset_game(RESET_LEVEL, INITIAL_LIVES, 0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array, key: chex.PRNGKey) -> tuple[
        PacmanObservation, PacmanState, jax.Array, jax.Array, PacmanInfo]:
        """
        Updates the game state by applying the game logic on the current state.
        """
        ( # 1) If in death animation, decrement timer and freeze everything
            new_state,
            frozen,
            done
        ) = self.death_step(state)
        
        ( # 2) Pacman handling
            player_position,
            player_action,
            pellets,
            has_pellet,
            collected_pellets,
            power_pellets,
            ate_power_pellet,
            pellet_reward,
            level_id
        ) = self.player_step(state, action)

        ( # 3) Fruit handling
            fruit_state,
            fruit_reward
        ) = self.fruit_step(state, player_position, collected_pellets, key)

        ( # 4) Ghost handling
            ghost_positions,
            ghost_actions,
            ghost_modes,
            ghost_timers,
            eaten_ghosts,
            new_lives,
            new_death_timer,
            ghosts_reward
        ) = self.ghosts_step(state, ate_power_pellet, key)

        # 5) Calculate reward, new score and flag score change digit-wise
        reward = pellet_reward + fruit_reward + ghosts_reward
        new_score = state.score + reward
        score_changed = self.flag_score_change(state.score, new_score)

        # 6) Update state
        new_state = jax.lax.cond(
            frozen,
            lambda: new_state,
            lambda: jax.lax.cond(
                level_id != state.level.id,
                lambda: reset_game(level_id, state.lives, new_score),
                lambda: PacmanState(
                    level = LevelState(
                        id=level_id,
                        collected_pellets=collected_pellets,
                        dofmaze=state.level.dofmaze,
                        pellets=pellets,
                        power_pellets=power_pellets,
                        loaded=jax.lax.cond(
                            state.level.loaded < 2,
                            lambda: state.level.loaded + 1,
                            lambda: state.level.loaded
                        )
                    ),
                    player = PlayerState(
                        position=player_position,
                        action=player_action,
                        has_pellet=has_pellet,
                        eaten_ghosts=eaten_ghosts
                    ),
                    ghosts = GhostsState(
                        positions=ghost_positions,
                        types=state.ghosts.types,
                        actions=ghost_actions,
                        modes=ghost_modes,
                        timers=ghost_timers,
                        keys=state.ghosts.keys
                    ),
                    fruit = fruit_state,
                    lives=new_lives,
                    score=new_score,
                    score_changed=score_changed,
                    freeze_timer=new_death_timer,
                    step_count=state.step_count + 1
                )
            )
        )

        # 7) Get observation, info and reward
        observation = self.get_observation(new_state)
        info = self.get_info(new_state)
        reward = jax.lax.cond(frozen, lambda: (0.0), lambda: (reward))
        return observation, new_state, reward, done, info
    
    @staticmethod
    @jax.jit
    def get_observation(state: PacmanState):
        return PacmanObservation(
            player_position=state.player.position,
            player_action=state.player.action,
            ghost_positions=state.ghosts.positions,
            ghost_actions=state.ghosts.actions,
            ghost_modes=state.ghosts.modes,
            ghosts_eaten=state.player.eaten_ghosts,
            fruit_position=state.fruit.position,
            fruit_action=state.fruit.position,
            fruit_type=state.fruit.type,
            pellets=state.level.pellets,
            power_pellets=state.level.power_pellets
        )

    @staticmethod
    @jax.jit
    def get_info(state: PacmanState):
        return PacmanInfo(
            level=state.level.id,
            score=state.score,
            lives=state.lives
        )
    
    @staticmethod
    def death_step(state: PacmanState):
        """
        Updates the game state when a deadly collision occured.
        """
        def decrement_timer(state: PacmanState):
            return state._replace(freeze_timer=state.freeze_timer - 1)

        return jax.lax.cond(
            state.freeze_timer == 0,
            lambda: (state, False, False), # Alive
            lambda: jax.lax.cond(
                state.freeze_timer > 1,
                lambda: (decrement_timer(state), True, False), # Frozen
                lambda: jax.lax.cond(
                    state.lives < 0,
                    lambda: (decrement_timer(state), True, True), # Game Over,
                    lambda: (reset_entities(decrement_timer(state)), True, False) # Level Reset
                )
            )
        )

    @staticmethod
    def player_step(state: PacmanState, action: chex.Array):
        """
        Updates the players position and orientation based on his input and the current maze layout.
        """
        # 1) Determine the last pressed action and check for validity
        action = last_pressed_action(action, state.player.action)
        action = jax.lax.cond(
            (action < 0) | (action > len(ACTIONS) - 1),
            lambda: Action.NOOP, # Ignore illegal actions
            lambda: action
        )
        # 2) Determine the next action based on the available directions
        available = available_directions(state.player.position, state.level.dofmaze)
        new_action = jax.lax.cond(
            (action != Action.NOOP) & (action != Action.FIRE) & available[action - 2],
            lambda: action,
            lambda: jax.lax.cond(
                (state.player.action > 1) & stop_wall(state.player.position, state.level.dofmaze)[state.player.action - 2],
                lambda: Action.NOOP,
                lambda: state.player.action
            )
        )
        # 3) Compute the next position 
        new_pos = state.player.position + ACTIONS[new_action]
        new_pos = new_pos.at[0].set(new_pos[0] % 160)
        # 4) Update pellets based on the new player position
        (
            pellets,
            has_pellet,
            collected_pellets,
            power_pellets,
            ate_power_pellet,
            reward,
            level_id
        ) = JaxPacman.pellet_step(state, new_pos)
        # 5) Return new player and pellet state
        return (
            new_pos,
            new_action,
            pellets,
            has_pellet,
            collected_pellets,
            power_pellets,
            ate_power_pellet,
            reward,
            level_id
        )

    @staticmethod
    def pellet_step(state: PacmanState, new_pacman_pos: chex.Array):
        """
        Updates pellets based on the players position and applies resulting score and mode changes.
        """
        def check_power_pellet(idx: chex.Array, power_pellets: chex.Array):
            return jax.lax.cond(
                idx < 0,
                lambda: False,
                lambda: power_pellets[idx % 4]
            )
        
        def eat_power_pellet(idx: chex.Array, power_pellets: chex.Array):
            return power_pellets.at[idx % 4].set(False)
        
        def check_pellet(pos: chex.Array):
            x_offset = jax.lax.cond(pos[0] < 75, lambda: 5, lambda: 1)
            return pos[0] % 8 == x_offset and pos[1] % 12 == 6
            
        def eat_pellet(pos: chex.Array, pellets: chex.Array):
            tile = (pos[0] - 2) // 8, (pos[1] + 4) // 12
            return jax.lax.cond(
                pellets[tile],
                lambda: (pellets.at[tile].set(False), True),
                lambda: (pellets, False)
            )
        
        # 1) Check if a regular pellet was eaten
        pellets, has_pellet = jax.lax.cond(
            check_pellet(new_pacman_pos),
            lambda: eat_pellet(new_pacman_pos, state.level.pellets),
            lambda: (state.level.pellets, False)
        )
        # 2) Check if a power pellet was eaten
        power_pellet_hit = jnp.where(
            jnp.all(jnp.round(new_pacman_pos / MsPacmanMaze.TILE_SCALE) == POWER_PELLET_HITBOXES, axis=1),
            size=1,
            fill_value=-1
        )[0].item()
        power_pellets, ate_power_pellet = jax.lax.cond(
            check_power_pellet(power_pellet_hit, state.level.power_pellets),
            lambda: (eat_power_pellet(power_pellet_hit, state.level.power_pellets), True),
            lambda: (state.level.power_pellets, False)
        )
        # 3) Process regular pellet reward
        collected_pellets, reward = jax.lax.cond(
            has_pellet,
            lambda: (state.level.collected_pellets + 1, PELLET_POINTS),
            lambda: (state.level.collected_pellets, 0)
        )
        # 4) Process power pellet reward
        reward = jax.lax.cond(
            ate_power_pellet,
            lambda: reward + POWER_PELLET_POINTS,
            lambda: reward
        )
        # 5) Check win condition
        level_id, reward = jax.lax.cond(
            collected_pellets >= PELLETS_TO_COLLECT,
            lambda: (state.level.id + 1, reward + LEVEL_COMPLETED_POINTS),
            lambda: (state.level.id, reward)
        )
        # 6) Update pellet state
        return (
            pellets,
            has_pellet,
            collected_pellets,
            power_pellets,
            ate_power_pellet,
            reward,
            level_id
        )

    @staticmethod
    def ghosts_step(state: PacmanState, ate_power_pellet: chex.Array, common_key: chex.Array
                    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Updates all ghosts.
        'key' can be a PRNGKey or None for deterministic behaviour.
        """
        new_positions = []
        new_actions = []
        new_modes = []
        new_timers = []

        # 1) Iterate over all ghosts
        for i in range(len(state.ghosts.types)):
            position    = state.ghosts.positions[i]
            action      = state.ghosts.actions[i]
            mode        = state.ghosts.modes[i]
            last_timer  = state.ghosts.timers[i]
            new_timer   = state.ghosts.timers[i]
            skip        = False

            # 2) Update ghost mode and timer
            if last_timer > 0:
                new_timer = last_timer - 1
            if ate_power_pellet and mode not in [GhostMode.ENJAILED, GhostMode.RETURNING]:
                mode = GhostMode.FRIGHTENED
                new_timer = FRIGHTENED_DURATION
                action = reverse_direction(action)
                skip = True
            elif new_timer == 0 and last_timer > 0:
                match mode:
                    case GhostMode.CHASE:
                        mode = GhostMode.SCATTER
                        new_timer = SCATTER_DURATION + jax.random.randint(common_key, (), 0, MAX_SCATTER_OFFSET)
                        action = reverse_direction(action)
                        skip = True
                    case GhostMode.SCATTER:
                        mode = GhostMode.CHASE
                        new_timer = CHASE_DURATION + jax.random.randint(common_key, (), 0, MAX_CHASE_OFFSET)
                        action = reverse_direction(action)
                        skip = True
                    case GhostMode.ENJAILED:
                        mode = GhostMode.RETURNING
                        new_timer = RETURN_DURATION
                        action = Action.UP
                        skip = True
                    case GhostMode.FRIGHTENED:
                        mode = GhostMode.BLINKING
                        new_timer = BLINKING_DURATION
                    case GhostMode.BLINKING | GhostMode.RETURNING | GhostMode.RANDOM:
                        mode = GhostMode.CHASE
                        new_timer = CHASE_DURATION

            # 3) Update ghost action
            if skip or mode == GhostMode.ENJAILED or mode == GhostMode.RETURNING:
                pass
            else:
                # Choose new direction
                allowed = get_allowed_directions(position, action, state.level.dofmaze)
                if not allowed: # If no direction is allowed - continue forward
                    pass
                elif len(allowed) == 1: # If only one allowed direction - take it
                    action = allowed[0]
                elif mode == GhostMode.FRIGHTENED or mode == GhostMode.BLINKING or mode == GhostMode.RANDOM or mode == GhostMode.RETURNING:
                    action = jax.random.choice(state.ghosts.keys[i], jnp.array(allowed))
                else:
                    if mode == GhostMode.CHASE:
                        chase_target = get_chase_target(state.ghosts.types[i], position, state.ghosts.positions[GhostType.BLINKY],
                                                        state.player.position, state.player.action)
                    else: # If not CHASE, mode must be SCATTER at this point
                        chase_target = SCATTER_TARGETS[state.ghosts.types[i]]
                    action = pathfind(position, action, chase_target, allowed, state.ghosts.keys[i])

            # 4) Update ghost position
            position = position + ACTIONS[action]
            position = position.at[0].set(position[0] % 160) # wrap horizontally

            # 5) Save new ghost state
            new_positions.append(position)
            new_actions.append(action)
            new_modes.append(mode)
            new_timers.append(new_timer)

        # 6) Check for player collision
        (
            new_positions,
            new_actions,
            new_modes,
            new_timers,
            eaten_ghosts,
            new_lives,
            new_death_timer,
            reward
        ) = JaxPacman.ghosts_collision(
            jnp.stack(new_positions),
            jnp.stack(new_actions),
            jnp.array(new_modes),
            jnp.array(new_timers),
            state.player.position,
            state.player.eaten_ghosts,
            ate_power_pellet,
            state.lives)

        return (
            new_positions,
            new_actions,
            new_modes,
            new_timers,
            eaten_ghosts,
            new_lives,
            new_death_timer,
            reward
        )

    @staticmethod
    def ghosts_collision(ghost_positions: chex.Array, ghost_actions: chex.Array, ghost_modes: chex.Array, ghost_timers: chex.Array,
                                new_pacman_pos: chex.Array, eaten_ghosts: chex.Array, ate_power_pellet: chex.Array, lives: chex.Array):
        """
        Updates the game state if a player-ghost collision occured.
        """
        deadly_collision = False
        reward = 0
        if ate_power_pellet:
            eaten_ghosts = 0
        for i in range(4): 
            if detect_collision(new_pacman_pos, ghost_positions[i]):
                if ghost_modes[i] == GhostMode.FRIGHTENED or ghost_modes[i] == GhostMode.BLINKING:  # If are frighted
                    # Ghost eaten
                    reward = EAT_GHOSTS_BASE_POINTS * (2 ** eaten_ghosts)
                    ghost_positions = ghost_positions.at[i].set(JAIL_POSITION)  # Reset eaten ghost position
                    ghost_actions = ghost_actions.at[i].set(Action.NOOP)  # Reset eaten ghost action
                    ghost_modes = ghost_modes.at[i].set(GhostMode.ENJAILED.value)
                    ghost_timers = ghost_timers.at[i].set(ENJAILED_DURATION)
                    eaten_ghosts = eaten_ghosts + 1
                else:
                    deadly_collision = True  # Collision with an already eaten and respawned ghost           
        else:
            deadly_collision = jnp.any(jnp.all(abs(new_pacman_pos - ghost_positions) < 8, axis=1))
        new_lives = lives - jnp.where(deadly_collision, 1, 0)
        new_death_timer = jnp.where(deadly_collision, RESET_TIMER, 0)
        return ghost_positions, ghost_actions, ghost_modes, ghost_timers, eaten_ghosts, new_lives, new_death_timer, reward

    @staticmethod
    def fruit_move(state: PacmanState, key: chex.Array
                   ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Updates the fruits position, action and timer if one is currently active.
        """
        allowed = get_allowed_directions(state.fruit.position, state.fruit.action, state.level.dofmaze)
        if not allowed:
            new_dir = state.fruit.action
        if len(allowed) == 1:
            new_dir = allowed[0]
        elif state.fruit.timer == 0:
            new_dir = pathfind(state.fruit.position, state.fruit.action, state.fruit.exit, allowed, key)
        else:
            new_dir = jax.random.choice(key, jnp.array(allowed))

        new_timer = state.fruit.timer
        if state.fruit.timer > 0:
            new_timer -= 1
        new_pos = jnp.array(state.fruit.position) + ACTIONS[new_dir]
        new_pos = new_pos.at[0].set(new_pos[0] % 160) # wrap horizontally
        return new_pos, new_dir, new_timer

    @staticmethod
    def fruit_step(state: PacmanState, new_pacman_pos: chex.Array, collected_pellets: chex.Array, key: chex.Array):
        """
        Updates the fruit state if a fruit spawns, moves or is consumed.
        """
        fruit_type = state.fruit.type
        fruit_position = state.fruit.position
        fruit_action = state.fruit.action
        fruit_timer = state.fruit.timer
        fruit_exit = state.fruit.exit
        reward = 0
        for threshold in FRUIT_SPAWN_THRESHOLDS:
            if collected_pellets == threshold and state.fruit.type == FruitType.NONE: # Spawn fruit
                fruit_type = get_level_fruit(state.level.id, key)
                fruit_position, fruit_action = get_random_tunnel(state.level.id, key)
                fruit_exit, _ = get_random_tunnel(state.level.id, key)
        if state.fruit.type != FruitType.NONE:
            if detect_collision(new_pacman_pos, state.fruit.position): # Consume fruit
                reward = FRUIT_REWARDS[state.fruit.type]
                fruit_type = FruitType.NONE
                fruit_timer = jnp.array(FRUIT_WANDER_DURATION).astype(jnp.uint8)
            if state.fruit.timer == 0 and jnp.all(jnp.array(state.fruit.position) == jnp.array(state.fruit.exit)): # Remove fruit
                fruit_type = FruitType.NONE
                fruit_timer = jnp.array(FRUIT_WANDER_DURATION).astype(jnp.uint8)
            else:
                fruit_position, fruit_action, fruit_timer = JaxPacman.fruit_move(state, key) # Move fruit
        return FruitState(fruit_position, fruit_exit, fruit_type, fruit_action, fruit_timer), reward

    @staticmethod
    def flag_score_change(current_score: chex.Array, new_score: chex.Array):
        """
        Flags the score digits for rendering that changed during the current step.
        """
        if new_score != current_score:
            score_digits        = aj.int_to_digits(new_score, max_digits=MAX_SCORE_DIGITS)
            state_score_digits  = aj.int_to_digits(current_score, max_digits=MAX_SCORE_DIGITS)
            score_changed       = score_digits != state_score_digits
        else:
            score_changed       = jnp.array(False, dtype=jnp.bool_)
        return score_changed


# -------- Render class --------
class MsPacmanRenderer(AtraJaxisRenderer):
    """JAX-based MsPacman game renderer, optimized with JIT compilation."""

    def __init__(self):
        super().__init__()
        self.sprites = MsPacmanRenderer.load_sprites()

    def render_background(self, level: chex.Array, lives: chex.Array, score: chex.Array):
        """Reset the background for a new level."""
        self.SPRITE_BG = MsPacmanMaze.load_background(get_level_maze(level))
        self.SPRITE_BG = MsPacmanRenderer.render_lives(self.SPRITE_BG, lives, self.sprites["pacman"][1][1]) # Life sprite (right looking pacman)
        self.SPRITE_BG = MsPacmanRenderer.render_score(self.SPRITE_BG, score, jnp.arange(MAX_SCORE_DIGITS) >= (MAX_SCORE_DIGITS - get_digit_count(score)), self.sprites["score"])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PacmanState):
        """Renders the current game state on screen."""
        # Render background for new game or level
        if state.level.loaded < 2:
            self.render_background(state.level.id, state.lives, state.score) # Render game over screen
        raster = self.SPRITE_BG

        # De-render pellets when consumed
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
                pellet_x, pellet_y = ((POWER_PELLET_TILES[pel_n][0] + 1) * MsPacmanMaze.TILE_SCALE,
                                      (POWER_PELLET_TILES[pel_n][1] + 2) * MsPacmanMaze.TILE_SCALE)
                raster = aj.render_at(raster, pellet_x, pellet_y, POWER_PELLET_SPRITE)
        # Render pacman
        orientation = state.player.action - 2 # convert action to direction
        pacman_sprite = self.sprites["pacman"][orientation][((state.step_count & 0b1000) >> 2)]
        raster = aj.render_at(raster, state.player.position[0], state.player.position[1], 
                              pacman_sprite)
        ghosts_orientation = ((state.step_count & 0b10000) >> 4) # (state.step_count % 32) // 16

        for i in range(len(state.ghosts.types)):
            # Render frightened ghost
            if not (state.ghosts.modes[i] == GhostMode.FRIGHTENED or state.ghosts.modes[i] == GhostMode.BLINKING):
                g_sprite = self.sprites["ghost"][ghosts_orientation][i]
            elif state.ghosts.modes[i] == GhostMode.BLINKING and ((state.step_count & 0b1000) >> 3):
                g_sprite = self.sprites["ghost"][ghosts_orientation][5] # white blinking effect
            else:
                g_sprite = self.sprites["ghost"][ghosts_orientation][4] # blue ghost
            raster = aj.render_at(raster, state.ghosts.positions[i][0], state.ghosts.positions[i][1], g_sprite)

        # Render fruit if present
        if state.fruit.type != FruitType.NONE:
            raster = MsPacmanRenderer.render_fruit(raster, state.fruit, self.sprites["fruit"])

        # Render score if changed
        if jnp.any(state.score_changed):
            self.SPRITE_BG = MsPacmanRenderer.render_score(self.SPRITE_BG, state.score, state.score_changed, self.sprites["score"])

        # Remove one life if a life is lost
        if state.freeze_timer == RESET_TIMER-1:
            self.SPRITE_BG = MsPacmanRenderer.render_lives(self.SPRITE_BG, state.lives, self.sprites["pacman"][1][1])
        return raster
    
    @staticmethod
    def render_score(raster, score, score_changed, digit_sprites, score_x=60, score_y=190, spacing=1, bg_color=jnp.array([0, 0, 0])):
        """Render the score on the raster at a fixed position. Only updates digits that have changed."""
        digits = aj.int_to_digits(score, max_digits=MAX_SCORE_DIGITS)
        for idx in range(len(digits)):
            if score_changed[idx]: 
                d_sprite    = digit_sprites[digits[idx]]
                bg_sprite   = jnp.full(d_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
                raster      = aj.render_at(raster, score_x + idx * (d_sprite.shape[1] + spacing), score_y, bg_sprite)
                raster      = aj.render_at(raster, score_x + idx * (d_sprite.shape[1] + spacing), score_y, d_sprite)
        return raster

    @staticmethod
    def render_lives(raster, current_lives, life_sprite, life_x=12, life_y=182, spacing=4, bg_color=jnp.array([0, 0, 0])):
        """Render the lives on the raster at a fixed position."""
        bg_sprite = jnp.full(life_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
        for i in range(MAX_LIVE_COUNT):
            if i < current_lives:
                raster = aj.render_at(raster, life_x + i * (life_sprite.shape[1] + spacing), life_y, life_sprite)
            else:
                raster = aj.render_at(raster, life_x + current_lives * (life_sprite.shape[1] + spacing), life_y, bg_sprite)            
        return raster
    
    @staticmethod
    def render_fruit(raster, fruit: FruitState, fruit_sprites):
        """Renders the fruit at its current position."""
        raster = aj.render_at(raster, fruit.position[0], fruit.position[1], fruit_sprites[fruit.type])
        return raster
    
    @staticmethod
    def load_sprites() -> dict[str, Any]:
        """Loads the game sprites from files into a class dictionary for rendering."""
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


# -------- Helper functions --------
def to_hashable(value):
    """Converts a value to a hashable type (e.g. from chex.Array to int or float)."""
    if hasattr(value, "item"):
        return value.item()
    return value


def get_digit_count(number: chex.Array) -> int:
    """Returns the number of digits in a given decimal number."""
    return len(str(abs(number)))


def last_pressed_action(action, prev_action):
    """Returns the last pressed action in cases where both actions are pressed"""
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
    """Degree of freedom of the object, can it move up, right, left, down"""
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
    """
    What directions are blocked for Pacman or the ghosts when at an intersection.
    Returns a tuple of booleans (up, right, left, down) indicating if
    the direction is blocked by a wall.

    Arguments:
    pos -- (x, y) position of the character
    dofmaze -- precomputed degree of freedom for a maze level/layout

    Returns:
    A tuple of booleans (up, right, left, down) indicating if that
    direction is blocked by a wall.
    """
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    up, right, left, down = dof(pos, dofmaze)
    return not(up) and on_horizontal_grid, not(right) and on_vertical_grid, not(left) and on_vertical_grid, not(down) and on_horizontal_grid


def get_allowed_directions(position: chex.Array, direction: chex.Array, dofmaze: chex.Array):
    """
    Returns an array of all directions (indices) in which movement is possible.
    Turning is only allowed at the centre of each tile and reverting is not allowed.
    """
    allowed = []
    if position[0] % 4 == 1 and position[1] % 12 == 6: # on horizontal and vertical grid - tile centre
        possible = available_directions(position, dofmaze) 
        for i, can_go in zip(DIRECTIONS, possible):
            if can_go and (direction == 0 or i != reverse_direction(direction)):
                allowed.append(i)
    else:
        allowed.append(direction)
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
            return player_pos + 4*MsPacmanMaze.TILE_SCALE * ACTIONS[player_dir]
        case GhostType.INKY:
            # Target the tip of the vector from Blinky to two tiles ahead of Pac-Man, doubled
            two_ahead = player_pos + 2*MsPacmanMaze.TILE_SCALE * ACTIONS[player_dir]
            vect = two_ahead - blinky_pos
            return blinky_pos + 2 * vect
        case GhostType.SUE:
            # Target Pac-Man if >8 tiles away, else target corner
            dist = jnp.linalg.norm(ghost_position - player_pos)
            return jnp.where(dist > 8*MsPacmanMaze.TILE_SCALE, player_pos, SCATTER_TARGETS[GhostType.SUE])


def pathfind(position: chex.Array, direction: chex.Array, target: chex.Array, allowed: chex.Array, key: chex.Array):
    """
    Returns the direction which should be taken to approach the target.
    If multiple options exist the direction is chosen that minimizes the distance on the longer axis - horizontal or vertical.
    If both distances are equal or multiple options exist on the same axis, the direction is chosen randomly.
    """
    # Check allowed directions
    if len(allowed) == 0: # If no direction allowed - Continue forward
        return direction
    if len(allowed) == 1: # If one direction allowed - Take it
        return allowed[0]

    # If multiple directions allowed - Get cost of allowed directions
    cost = {}
    for dir in allowed:
        new_pos = position + ACTIONS[dir]
        cost[to_hashable(dir)] = jnp.abs(new_pos[0] - target[0]) + jnp.abs(new_pos[1] - target[1])
    min_cost = min(cost.values())
    min_dirs = jnp.array([k for k, v in cost.items() if v == min_cost])
    if len(min_dirs) == 1: # If one direction advantageous - Take it
        return min_dirs[0]
    
    # If multiple directions advantageous - Prioritize the longer axis
    horizontal_distance = jnp.abs(position[0] - target[0])
    vertical_distance = jnp.abs(position[1] - target[1])
    mask = jnp.full(len(min_dirs), False)
    if horizontal_distance >= vertical_distance:
        mask |= jnp.isin(min_dirs, jnp.array([Action.LEFT, Action.RIGHT]))
    if vertical_distance >= horizontal_distance:
        mask |= jnp.isin(min_dirs, jnp.array([Action.DOWN, Action.UP]))

    if jnp.sum(mask) == 0: # If no direction advantageous on longer axis - Choose randomly
        return jax.random.choice(key, min_dirs)
    if jnp.sum(mask) == 1: # If one direction advantageous on longer axis - Take it
        return jnp.squeeze(min_dirs[mask])
    else: # If multiple directions advantageous on longer or equal axis - Choose randomly with mask
        return jax.random.choice(key, min_dirs[mask])


def get_level_maze(level: chex.Array):
    """Returns the maze id that correpsonds to the current level."""
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
    """Returns the fruit that corresponds to the current level."""
    match level:
        case 1: return FruitType.CHERRY
        case 2: return FruitType.STRAWBERRY
        case 3: return FruitType.ORANGE
        case 4: return FruitType.PRETZEL
        case 5: return FruitType.APPLE
        case 6: return FruitType.PEAR
        case 7: return FruitType.BANANA
        case _: return jax.random.randint(key, (), 1, 8)


def get_random_tunnel(level: chex.Array, key: chex.Array):
    """Returns the position and exit direction of a random tunnel."""
    maze = get_level_maze(level)
    tunnel_heights = MsPacmanMaze.TUNNEL_HEIGHTS[maze]
    tunnels = [[0, tunnel_heights[0], Action.RIGHT], [MsPacmanMaze.WIDTH - 1, tunnel_heights[0], Action.LEFT], 
               [0, tunnel_heights[1], Action.RIGHT], [MsPacmanMaze.WIDTH - 1, tunnel_heights[1], Action.LEFT]]
    if tunnel_heights[1] == 0: # If the second element is 0, there is only one pair of tunnels
        tunnel_idx = jax.random.randint(key, (), 0, 2)
    else:
        tunnel_idx = jax.random.randint(key, (), 0, 4)

    tunnel = tunnels[tunnel_idx]
    tunnel_pos = tunnel[:2]
    tunnel_dir = tunnel[2]
    return tunnel_pos, tunnel_dir


def reverse_direction(dir_idx: chex.Array):
    """Inverts the direction if possible."""
    INV_DIR = {2:5, 3:4, 4:3, 5:2}
    dir_idx = to_hashable(dir_idx)
    if dir_idx not in INV_DIR:
        return dir_idx
    return INV_DIR[dir_idx]
    

def detect_collision(position_1: chex.Array, position_2: chex.Array):
    """Checks if the two positions are closer than the collision threshold."""
    if jnp.all(abs(jnp.array(position_1) - jnp.array(position_2)) < COLLISION_THRESHOLD):
        return True
    return False


# -------- Reset functions --------
def reset_level(level: chex.Array):
    maze = MsPacmanMaze.MAZES[get_level_maze(level)]
    return LevelState(
        id                  = jnp.array(level, dtype=jnp.uint8),
        collected_pellets   = jnp.array(0).astype(jnp.uint8),
        dofmaze             = MsPacmanMaze.precompute_dof(maze), # Precompute degree of freedom maze layout
        pellets             = jnp.copy(MsPacmanMaze.BASE_PELLETS),
        power_pellets       = jnp.ones(4, dtype=jnp.bool_),
        loaded              = jnp.array(0, dtype=jnp.uint8)
    )

def reset_player():
    return PlayerState(
        position            = INITIAL_PACMAN_POSITION,
        action              = INITIAL_ACTION,
        has_pellet          = jnp.array(False),
        eaten_ghosts        = jnp.array(0).astype(jnp.uint8) # number of eaten ghost since power pellet consumed
    )

def reset_ghosts():
    seed = int(time.time() * 1000) % (2**32 - 1)
    base_key = jax.random.PRNGKey(seed)
    unique_keys = jax.random.split(base_key, 4)
    return GhostsState (
        positions   = INITIAL_GHOSTS_POSITIONS,
        types       = jnp.array([GhostType.BLINKY, GhostType.PINKY, GhostType.INKY, GhostType.SUE], dtype=jnp.uint8),
        actions     = jnp.array([Action.NOOP, Action.NOOP, Action.NOOP, Action.NOOP], dtype=jnp.uint8),
        modes       = jnp.array([GhostMode.RANDOM, GhostMode.RANDOM, GhostMode.SCATTER, GhostMode.SCATTER], dtype=jnp.uint8),
        timers      = jnp.array([SCATTER_DURATION, SCATTER_DURATION, SCATTER_DURATION, SCATTER_DURATION], dtype=jnp.uint16),
        keys        = unique_keys
    )

def reset_fruit():
    return FruitState(
        position    = jnp.zeros(2, dtype=jnp.int8),
        exit        = jnp.zeros(2, dtype=jnp.int8),
        type        = jnp.array(FruitType.NONE).astype(jnp.uint8),
        action      = jnp.array(Action.NOOP).astype(jnp.uint8),
        timer       = jnp.array(FRUIT_WANDER_DURATION).astype(jnp.uint8)
    )

def reset_game(level: chex.Array, lives: chex.Array, score: chex.Array):
    return PacmanState(
        level           = reset_level(level),
        player          = reset_player(),
        ghosts          = reset_ghosts(),
        fruit           = reset_fruit(),
        lives           = jnp.array(lives, dtype=jnp.int8),
        score           = jnp.array(score, dtype=jnp.uint32),
        score_changed   = jnp.arange(MAX_SCORE_DIGITS) >= (MAX_SCORE_DIGITS - get_digit_count(score)),
        freeze_timer    = jnp.array(0, dtype=jnp.uint32),
        step_count      = jnp.array(0, dtype=jnp.uint32),
    )

def reset_entities(state: PacmanState):
    return PacmanState(
        level = LevelState(
            id = state.level.id,
            collected_pellets=state.level.collected_pellets,
            dofmaze=state.level.dofmaze,
            pellets=state.level.pellets,
            power_pellets=state.level.power_pellets,
            loaded=state.level.loaded
        ),
        player          = reset_player(),
        ghosts          = reset_ghosts(),
        fruit           = reset_fruit(),
        lives           = state.lives,
        score           = state.score,
        score_changed   = state.score_changed,
        freeze_timer    = state.freeze_timer,
        step_count      = state.step_count,
    )
