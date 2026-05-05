# -------- Imports --------
from enum import IntEnum
import os
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
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
class MsPacmanConstants(struct.PyTreeNode):
    # GENERAL
    RESET_LEVEL: int = struct.field(pytree_node=False, default=1) # The starting level, loaded when reset is called
    TIME_SCALE: int = struct.field(pytree_node=False, default=20) # Approximate number of timesteps in a second scaled to the original game speed
    INITIAL_LIVES: int = struct.field(pytree_node=False, default=3) # Number of starting bonus lives
    MAX_LIVE_COUNT: int = struct.field(pytree_node=False, default=4) # Maximum possible number of lives
    MAX_SCORE_DIGITS: int = struct.field(pytree_node=False, default=6) # Number of digits to display in the score
    BONUS_LIFE_SCORE: int = struct.field(pytree_node=False, default=10000) # Score at which a bonus life is rewarded
    COLLISION_THRESHOLD: int = struct.field(pytree_node=False, default=6) # Contacts below this distance count as collision
    PELLETS_TO_COLLECT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([154, 150, 158, 154])) # Total pellets to collect in each maze
    DOF_MAZES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: MsPacmanMaze.get_dof_mazes())

    # GHOST TIMINGS
    SUE_RELEASE_TIME: int = struct.field(pytree_node=False, default=1*20)
    INKY_RELEASE_TIME: int = struct.field(pytree_node=False, default=5*20)
    PINKY_RELEASE_TIME: int = struct.field(pytree_node=False, default=7*20)
    RESET_TIMER: int = struct.field(pytree_node=False, default=4*20)
    CHASE_DURATION: int = struct.field(pytree_node=False, default=20*20)
    SCATTER_DURATION: int = struct.field(pytree_node=False, default=7*20)
    FRIGHTENED_DURATION: int = struct.field(pytree_node=False, default=13*20)
    BLINKING_DURATION: int = struct.field(pytree_node=False, default=4*20)
    ENJAILED_DURATION: int = struct.field(pytree_node=False, default=10*20)
    FRIGHTENED_REDUCTION: float = struct.field(pytree_node=False, default=0.85)
    RETURN_DURATION: int = struct.field(pytree_node=False, default=int(20/2))
    MAX_CHASE_OFFSET: float = struct.field(pytree_node=False, default=20*20/10)
    MAX_SCATTER_OFFSET: float = struct.field(pytree_node=False, default=7*20/10)

    # FRUITS
    FRUIT_SPAWN_THRESHOLDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([50, 100]))
    FRUIT_WANDER_DURATION: int = struct.field(pytree_node=False, default=25*20)

    # POSITIONS
    POWER_PELLET_TILES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[1, 3], [36, 3], [1, 36], [36, 36]]))
    POWER_PELLET_HITBOXES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[1, 3], [36, 3], [1, 36], [36, 36], [1, 4], [36, 4], [1, 37], [36, 37]]))
    JAIL_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([75, 75]))
    INITIAL_GHOSTS_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[75, 54], [75, 75], [75, 75], [75, 75]]))
    INITIAL_PACMAN_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([75, 102]))
    SCATTER_TARGETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[MsPacmanMaze.WIDTH - 1, 0], [0, 0], [MsPacmanMaze.WIDTH - 1, MsPacmanMaze.HEIGHT - 1], [0, MsPacmanMaze.HEIGHT - 1]]))

    # ACTIONS
    DIRECTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN]))
    ACTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([(0, 0), (0, 0), (0, -1), (1, 0), (-1, 0), (0, 1)]))
    INITIAL_ACTION: int = struct.field(pytree_node=False, default=Action.LEFT)
    INITIAL_LAST_ACTION: int = struct.field(pytree_node=False, default=Action.LEFT)

    # POINTS
    PELLET_POINTS: int = struct.field(pytree_node=False, default=10)
    POWER_PELLET_POINTS: int = struct.field(pytree_node=False, default=50)
    FRUIT_REWARDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([100, 200, 500, 700, 1000, 2000, 5000]))
    EAT_GHOSTS_BASE_POINTS: int = struct.field(pytree_node=False, default=200)
    LEVEL_COMPLETED_POINTS: int = struct.field(pytree_node=False, default=500)

    # COLORS
    PATH_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 28, 136], dtype=jnp.uint8))
    WALL_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([228, 111, 111], dtype=jnp.uint8))
    PELLET_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([228, 111, 111], dtype=jnp.uint8))
    PACMAN_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([210, 164, 74, 255], dtype=jnp.uint8))


# -------- Entity classes --------
class LevelState(NamedTuple):
    id: chex.Array                  # Int - Number of the current level, starts at 1
    collected_pellets: chex.Array   # Int - Number of collected pellets
    pellets: chex.Array             # Bool[x][y] - 2D grid of 0 (empty) or 1 (pellet)
    power_pellets: chex.Array       # Bool[4] - Indicates wheter the power pellet is available
    loaded: chex.Array              # Int - 0: Not loaded, 1: loading, 2: loaded

class GhostsState(NamedTuple):
    positions: chex.Array           # Tuple - (x, y)
    types: chex.Array               # Enum - 0: BLINKY, 1: PINKY, 2: INKY, 3: SUE
    actions: chex.Array             # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    modes: chex.Array               # Enum - 0: RANDOM, 1: CHASE, 2: SCTATTER, 3: FRIGHTENED, 4: BLINKING, 5: RETURNING, 6: ENJAILED
    timers: chex.Array              # Int - Triggers mode change when reaching 0, decrements every step

class PlayerState(NamedTuple):
    position: chex.Array            # Tuple - (x, y)
    action: chex.Array              # Enum - 0: NOOP, 1: FURE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    has_pellet: chex.Array          # Bool - Indicates if pacman just collected a pellet
    eaten_ghosts: chex.Array        # Int - Indicates the number of ghosts eaten since the last power pellet
    last_horiz_dir: chex.Array = jnp.array(2, dtype=jnp.int32) # Default LEFT (2 in act_to_dir)

class FruitState(NamedTuple):
    position: chex.Array            # Tuple - (x, y)
    exit: chex.Array                # Tuple - (x, y) Position of the tunnel through which it will exit
    type: chex.Array                # Enum - 0: CHERRY, 1: STRAWBERRY, 2: ORANGE, 3: PRETZEL, 4: APPLE, 5: PEAR, 6: BANANA, 7: NONE
    action: chex.Array              # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    spawn: chex.Array               # Bool - Indicates wether a fruit should spawn into the maze as soon as possible
    spawned: chex.Array             # Bool - Indicates wether a fruit is currently present within the maze
    timer: chex.Array               # Int - Time until leaving through the exit tunnel, decrements every step

@struct.dataclass
class PacmanState:
    level: LevelState               # LevelState
    player: PlayerState             # PlayerState
    ghosts: GhostsState             # GhostStates
    fruit: FruitState               # FruitState
    lives: chex.Array               # Int - Number of lives left
    score: chex.Array               # Int - Total score reached
    score_changed: chex.Array       # Bool[] - Indicates which score digit changed since the last step
    freeze_timer: chex.Array        # Int - Time until game is unfrozen, decrements every step
    step_count: chex.Array          # Int - Number of steps made in the current level
    key: chex.PRNGKey               # PRNGKey for RNG during step

@struct.dataclass
class PacmanObservation:
    player_position: chex.Array
    player_action: chex.Array
    ghost_positions: chex.Array
    ghost_actions: chex.Array
    fruit_position: chex.Array
    fruit_action: chex.Array
    fruit_type: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array

@struct.dataclass
class PacmanInfo:
    level: chex.Array
    score: chex.Array
    lives: chex.Array


# -------- Game class --------
class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, MsPacmanConstants]):
    def __init__(self, consts: MsPacmanConstants = None):
        consts = consts or MsPacmanConstants()
        super().__init__(consts)
        self.frame_stack_size = 1
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
        ]
        self.renderer = MsPacmanRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for MsPacman.
        Actions are:
        0: NOOP
        1: UP
        2: RIGHT
        3: LEFT
        4: DOWN
        5: UPRIGHT
        6: UPLEFT
        7: DOWNRIGHT
        8: DOWNLEFT
        """
        return spaces.Discrete(9)

    def reset(self, key=None) -> Tuple[PacmanObservation, PacmanState]:
        """
        Resets the game to its initial state.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        state = reset_game(self.consts, self.consts.RESET_LEVEL, self.consts.INITIAL_LIVES, 0, key)
        return self.get_observation(state), state

    def render(self, state: PacmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array) -> tuple[
        PacmanObservation, PacmanState, jax.Array, jax.Array, PacmanInfo]:
        """
        Updates the game state by applying the game logic on the current state.
        """
        key, step_key = jax.random.split(state.key)
        
        ( # 1) If in death animation, decrement timer and freeze everything
            new_state,
            frozen,
            done
        ) = self.death_step(state, step_key, self.consts)
        
        maze_idx = get_level_maze(state.level.id)
        dofmaze = self.consts.DOF_MAZES[maze_idx]

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
        ) = self.player_step(state, action, dofmaze, self.consts)

        ( # 3) Fruit handling
            fruit_state,
            fruit_reward
        ) = self.fruit_step(state, player_position, collected_pellets, dofmaze, step_key, self.consts)

        ( # 4) Ghost handling
            ghost_positions,
            ghost_actions,
            ghost_modes,
            ghost_timers,
            eaten_ghosts,
            new_lives,
            new_death_timer,
            ghosts_reward
        ) = self.ghosts_step(state, ate_power_pellet, dofmaze, step_key, self.consts)

        # 5) Calculate reward, new score, bonus life and flag score change digit-wise
        reward = pellet_reward + fruit_reward + ghosts_reward
        new_score = state.score + reward
        score_changed = self.flag_score_change(state.score, new_score, self.consts)
        new_lives = jax.lax.cond(
            (new_score >= self.consts.BONUS_LIFE_SCORE) & (state.score < self.consts.BONUS_LIFE_SCORE),
            lambda: new_lives + 1,
            lambda: new_lives
        )
        
        # 6) Update state
        new_state = jax.lax.cond(
            frozen,
            lambda: new_state.replace(key=key),
            lambda: jax.lax.cond(
                level_id != state.level.id,
                lambda: reset_game(self.consts, level_id, state.lives, new_score, key),
                lambda: PacmanState(
                    level = LevelState(
                        id=level_id,
                        collected_pellets=collected_pellets,
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
                        timers=ghost_timers
                    ),
                    fruit=fruit_state,
                    lives=new_lives,
                    score=new_score,
                    score_changed=score_changed,
                    freeze_timer=new_death_timer,
                    step_count=state.step_count + 1,
                    key=key
                )
            )
        )

        # 7) Get observation, info and reward
        observation = self.get_observation(new_state)
        info = self.get_info(new_state)
        reward = jax.lax.cond(
            frozen,
            lambda: jnp.array(0, dtype=jnp.uint32),
            lambda: jnp.array(reward, dtype=jnp.uint32)
        )
        return observation, new_state, reward, done, info
    
    @staticmethod
    @jax.jit
    def get_observation(state: PacmanState):
        return PacmanObservation(
            player_position=state.player.position,
            player_action=state.player.action,
            ghost_positions=state.ghosts.positions,
            ghost_actions=state.ghosts.actions,
            fruit_position=state.fruit.position,
            fruit_action=state.fruit.action,
            fruit_type=state.fruit.type,
            pellets=state.level.pellets.astype(jnp.uint8),
            power_pellets=state.level.power_pellets.astype(jnp.uint8)
        )

    @staticmethod
    @jax.jit
    def get_info(state: PacmanState):
        return PacmanInfo(
            level=state.level.id,
            score=state.score,
            lives=state.lives
        )

    def _get_observation(self, state: PacmanState) -> PacmanObservation:
        return JaxPacman.get_observation(state)

    def _get_info(self, state: PacmanState, all_rewards=None) -> PacmanInfo:
        return JaxPacman.get_info(state)

    def _get_reward(self, previous_state: PacmanState, state: PacmanState) -> chex.Array:
        return (state.score - previous_state.score).astype(jnp.float32)

    def _get_done(self, state: PacmanState) -> chex.Array:
        return state.lives < 0

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player_position": spaces.Box(low=0, high=255, shape=(2,), dtype=jnp.int32),
            "player_action": spaces.Box(low=0, high=8, shape=(), dtype=jnp.uint8),
            "ghost_positions": spaces.Box(low=0, high=255, shape=(4, 2), dtype=jnp.int32),
            "ghost_actions": spaces.Box(low=0, high=8, shape=(4,), dtype=jnp.uint8),
            "fruit_position": spaces.Box(low=0, high=255, shape=(2,), dtype=jnp.uint8),
            "fruit_action": spaces.Box(low=0, high=8, shape=(), dtype=jnp.uint8),
            "fruit_type": spaces.Box(low=0, high=6, shape=(), dtype=jnp.uint8),
            "pellets": spaces.Box(low=0, high=1, shape=(18, 14), dtype=jnp.uint8),
            "power_pellets": spaces.Box(low=0, high=1, shape=(4,), dtype=jnp.uint8),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @staticmethod
    def death_step(state: PacmanState, key: chex.PRNGKey, consts: MsPacmanConstants):
        """
        Updates the game state when a deadly collision occured.
        """
        def decrement_timer(state: PacmanState):
            return state.replace(freeze_timer=state.freeze_timer - 1)

        return jax.lax.cond(
            state.freeze_timer == 0,
            lambda: (state, False, False), # Alive
            lambda: jax.lax.cond(
                state.freeze_timer > 1,
                lambda: (decrement_timer(state), True, False), # Frozen
                lambda: jax.lax.cond(
                    state.lives == 0,
                    lambda: (decrement_timer(state), True, True), # Game Over,
                    lambda: (reset_entities(consts, decrement_timer(state), key), True, False) # Level Reset
                )
            )
        )

    @staticmethod
    def player_step(state: PacmanState, action: chex.Array, dofmaze: chex.Array, consts: MsPacmanConstants):
        """
        Updates the players position and orientation based on his input and the current maze layout.
        """
        # 1) Determine the last pressed action and check for validity
        action = last_pressed_action(action, state.player.action)
        action = jnp.asarray(action, dtype=state.player.action.dtype)
        action = jax.lax.cond(
            (action < 0) | (action > len(consts.ACTIONS) - 1),
            lambda: jnp.array(Action.NOOP, dtype=state.player.action.dtype), # Ignore illegal actions
            lambda: action
        )
        # 2) Determine the next action based on the available directions
        available = available_directions(state.player.position, dofmaze)
        new_action = jax.lax.cond(
            (action != Action.NOOP) & available[act_to_dir(action)],
            lambda: action,
            lambda: state.player.action
        )
        # 3) Compute the next position
        new_pos = jax.lax.cond(
            stop_wall(state.player.position, dofmaze)[act_to_dir(state.player.action)],
            lambda: state.player.position,
            lambda: get_new_position(state.player.position, new_action, consts)
        )
        # 4) Update pellets based on the new player position
        (
            pellets,
            has_pellet,
            collected_pellets,
            power_pellets,
            ate_power_pellet,
            reward,
            level_id
        ) = JaxPacman.pellet_step(state, new_pos, consts)
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
    def pellet_step(state: PacmanState, new_pacman_pos: chex.Array, consts: MsPacmanConstants):
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
            return (pos[0] % 8 == x_offset) & (pos[1] % 12 == 6)
            
        def eat_pellet(pos: chex.Array, pellets: chex.Array):
            tile_x, tile_y = (pos[0] - 2) // 8, (pos[1] + 4) // 12
            in_bounds = (tile_x >= 0) & (tile_x < pellets.shape[0]) & (tile_y >= 0) & (tile_y < pellets.shape[1])
            return jax.lax.cond(
                pellets[tile_x, tile_y] & in_bounds,
                lambda: (pellets.at[tile_x, tile_y].set(False), True),
                lambda: (pellets, False)
            )
        
        # 1) Check if a regular pellet was eaten
        pellets, ate_pellet = jax.lax.cond(
            check_pellet(new_pacman_pos),
            lambda: eat_pellet(new_pacman_pos, state.level.pellets),
            lambda: (state.level.pellets, False)
        )
        # 2) Check if a power pellet was eaten
        # Optimized hit detection avoiding jnp.where(..., size=1)
        power_pellet_matches = jnp.all(jnp.round(new_pacman_pos / MsPacmanMaze.TILE_SCALE) == consts.POWER_PELLET_HITBOXES, axis=1)
        has_hit = jnp.any(power_pellet_matches)
        power_pellet_hit = jnp.where(has_hit, jnp.argmax(power_pellet_matches), -1)

        power_pellets, ate_power_pellet = jax.lax.cond(
            check_power_pellet(power_pellet_hit, state.level.power_pellets),
            lambda: (eat_power_pellet(power_pellet_hit, state.level.power_pellets), True),
            lambda: (state.level.power_pellets, False)
        )
        # 3) Process pellet reward
        reward = jax.lax.cond(
            ate_power_pellet,
            lambda: consts.POWER_PELLET_POINTS,
            lambda: jax.lax.cond(
                ate_pellet,
                lambda: consts.PELLET_POINTS,
                lambda: 0
            )
        )
        # 4) Update collected pellets
        has_pellet = ate_power_pellet | ate_pellet
        collected_pellets = jax.lax.cond(
            has_pellet,
            lambda: state.level.collected_pellets + 1,
            lambda: state.level.collected_pellets
        )
        # 5) Check win condition
        level_id, reward = jax.lax.cond(
            collected_pellets >= consts.PELLETS_TO_COLLECT[get_level_maze(state.level.id)],
            lambda: (state.level.id + 1, reward + consts.LEVEL_COMPLETED_POINTS),
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
    def ghosts_step(state: PacmanState, ate_power_pellet: chex.Array, dofmaze: chex.Array, common_key: chex.Array, consts: MsPacmanConstants
                    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Updates all ghosts and checks for collisions with the player.
        """
        ghost_keys  = jax.random.split(common_key, 4)

        def update_ghost_mode(mode, action, timer, step_count, ate_power_pellet):
            new_timer = jax.lax.cond(
                timer > 0,
                lambda: jnp.array(timer - 1.0, dtype=jnp.float16),
                lambda: jnp.array(timer, dtype=jnp.float16)
            )
            timing_factor = jax.lax.cond(
                state.level.id == 1,
                lambda: 1.0,
                lambda: consts.FRIGHTENED_REDUCTION ** (state.level.id - 1)
            )
            return jax.lax.cond(
                ate_power_pellet & (mode != GhostMode.ENJAILED) & (mode != GhostMode.RETURNING),
                lambda: (
                    jnp.array(GhostMode.FRIGHTENED, dtype=jnp.uint8),
                    jnp.array(reverse_action(action), dtype=jnp.uint8),
                    jnp.array(consts.FRIGHTENED_DURATION * timing_factor, dtype=jnp.float16),
                    True
                ),
                lambda: jax.lax.cond(
                    (timer > 0) & (new_timer <= 0),
                    lambda: jax.lax.switch(
                        mode,
                        (
                            start_chase_no_reverse, # 0: RANDOM
                            start_scatter,          # 1: CHASE
                            start_chase_offset,     # 2: SCATTER
                            start_blinking,         # 3: FRIGHTENED
                            start_chase_no_reverse, # 4: BLINKING
                            start_returned,         # 5: RETURNING
                            start_returning         # 6: ENJAILED
                        ),
                        action, step_count
                    ),
                    lambda: (
                        mode,
                        action,
                        new_timer,
                        False
                    )
                )
            )

        def start_scatter(action, step_count): # succeeds chase mode
            OFFSET_SCALE = 10.0
            scaled_offset = jnp.round(consts.MAX_SCATTER_OFFSET * OFFSET_SCALE)
            return (
                jnp.array(GhostMode.SCATTER, dtype=jnp.uint8),
                jnp.array(action, dtype=jnp.uint8),
                jnp.array(consts.SCATTER_DURATION + (jax.random.randint(common_key, (), -scaled_offset, scaled_offset) / OFFSET_SCALE), dtype=jnp.float16),
                False
            )
        
        def start_chase_offset(action, step_count): # succeeds scatter mode
            OFFSET_SCALE = 10.0
            scaled_offset = jnp.round(consts.MAX_CHASE_OFFSET * OFFSET_SCALE)
            return (
                jnp.array(GhostMode.CHASE, dtype=jnp.uint8),
                jnp.array(action, dtype=jnp.uint8),
                jnp.array(consts.CHASE_DURATION + (jax.random.randint(common_key, (), -scaled_offset, scaled_offset) / OFFSET_SCALE), dtype=jnp.float16),
                False
            )
        
        def start_chase_no_reverse(action, step_count): # succeeds blinking, returning and random mode
            return (
                jnp.array(GhostMode.CHASE, dtype=jnp.uint8),
                jnp.array(action, dtype=jnp.uint8),
                jnp.array(consts.CHASE_DURATION, dtype=jnp.float16),
                False
            )
        
        def start_returning(action, step_count): # succeeds enjailed mode
            return (
                jnp.array(GhostMode.RETURNING, dtype=jnp.uint8),
                jnp.array(Action.UP, dtype=jnp.uint8),
                jnp.array(consts.RETURN_DURATION, dtype=jnp.float16),
                True
            )
        
        def start_blinking(action, step_count): # succeeds frightened mode
            timing_factor = jax.lax.cond(
                state.level.id == 1,
                lambda: 1.0,
                lambda: consts.FRIGHTENED_REDUCTION ** (state.level.id - 1)
            )
            return (
                jnp.array(GhostMode.BLINKING, dtype=jnp.uint8),
                jnp.array(action, dtype=jnp.uint8),
                jnp.array(jnp.round(consts.BLINKING_DURATION * timing_factor), dtype=jnp.float16),
                False
            )
        
        def start_returned(action, step_count): # returning
            return jax.lax.cond(
                step_count > (consts.SCATTER_DURATION + 1) * consts.TIME_SCALE,
                lambda: start_chase_no_reverse(action, step_count),
                lambda: (
                    jnp.array(GhostMode.RANDOM, dtype=jnp.uint8),
                    jnp.array(action, dtype=jnp.uint8),
                    jnp.array(consts.SCATTER_DURATION, dtype=jnp.float16),
                    False
                )
            )

        def pathfind_target(type, mode, action, position, allowed, key):
            chase_target = jax.lax.cond(
                mode == GhostMode.CHASE,
                lambda: get_chase_target(type, position, state.ghosts.positions[GhostType.BLINKY], state.player.position, state.player.action, consts.ACTIONS, consts.SCATTER_TARGETS),
                lambda: consts.SCATTER_TARGETS[type]
            )
            return pathfind(position, action, chase_target, allowed, key, consts.ACTIONS, consts.DIRECTIONS)

        def choose_direction(type, mode, action, position, key):
            allowed = get_allowed_directions(position, action, dofmaze, consts.DIRECTIONS, is_ghost=True)
            n_allowed = jnp.sum(allowed != 0)
            return jax.lax.cond(
                n_allowed == 0,
                lambda: action,
                lambda: jax.lax.cond(
                    n_allowed == 1,
                    lambda: allowed[0],
                    lambda: jax.lax.cond(
                        (mode == GhostMode.FRIGHTENED) | (mode == GhostMode.BLINKING) | (mode == GhostMode.RANDOM) | (mode == GhostMode.RETURNING),
                        lambda: allowed[jax.random.randint(key, (), minval=0, maxval=n_allowed)],
                        lambda: pathfind_target(type, mode, action, position, allowed, key)
                    )
                )
            )

        def ghost_step_single(ghost_type, mode, action, position, timer, key):
            new_mode, new_action, new_timer, skip = update_ghost_mode(
                mode, action, timer, state.step_count, ate_power_pellet
            )

            new_action = jnp.asarray(
                jax.lax.cond(
                    skip | (new_mode == GhostMode.ENJAILED) | (new_mode == GhostMode.RETURNING),
                    lambda: new_action,
                    lambda: choose_direction(ghost_type, new_mode, new_action, position, key)
                ),
                dtype=jnp.uint8
            )

            new_position, new_action = jax.lax.cond(
                ((mode == GhostMode.FRIGHTENED) |
                 (mode == GhostMode.BLINKING) |
                 (mode == GhostMode.RETURNING)) &
                 (state.step_count % 2 == 0),
                lambda: (position, action),
                lambda: (get_new_position(position, new_action, consts), new_action)
            )
            return new_mode, new_action, new_position, new_timer

        vmapped_ghost_step = jax.vmap(ghost_step_single)
        new_modes, new_actions, new_positions, new_timers = vmapped_ghost_step(
            state.ghosts.types,
            state.ghosts.modes,
            state.ghosts.actions,
            state.ghosts.positions,
            state.ghosts.timers,
            ghost_keys
        )
        
        ( # Check for player collision
            new_positions,
            new_actions,
            new_modes,
            new_timers,
            eaten_ghosts,
            new_lives,
            new_death_timer,
            reward
        ) = JaxPacman.ghosts_collision(
            new_positions,
            new_actions,
            new_modes,
            new_timers,
            state.player.position,
            state.player.eaten_ghosts,
            ate_power_pellet,
            state.lives,
            consts)

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
                                new_pacman_pos: chex.Array, eaten_ghosts: chex.Array, ate_power_pellet: chex.Array, lives: chex.Array, consts: MsPacmanConstants):
        """
        Updates the game state if a player-ghost collision occured.
        """
        def handle_ghost_collision_single(ghost_index, ghost_pos, ghost_mode, ghost_action, ghost_timer, current_eaten_ghosts):
            collision = detect_collision(new_pacman_pos, ghost_pos, consts.COLLISION_THRESHOLD)
            
            is_frightened = (ghost_mode == GhostMode.FRIGHTENED) | (ghost_mode == GhostMode.BLINKING)
            
            # Case 1: Ghost eaten
            eaten_reward = consts.EAT_GHOSTS_BASE_POINTS * (2 ** current_eaten_ghosts.astype(jnp.uint32))
            
            # We need to know if THIS ghost was eaten to update its state
            this_ghost_eaten = collision & is_frightened
            
            new_pos = jnp.where(this_ghost_eaten, consts.JAIL_POSITION, ghost_pos)
            new_act = jnp.where(this_ghost_eaten, Action.NOOP, ghost_action)
            new_mode = jnp.where(this_ghost_eaten, GhostMode.ENJAILED.value, ghost_mode)
            new_timer = jnp.where(this_ghost_eaten, consts.ENJAILED_DURATION, ghost_timer)
            
            # Case 2: Deadly collision
            deadly = collision & (~is_frightened)
            
            return new_pos, new_act, new_mode, new_timer, this_ghost_eaten, deadly, eaten_reward

        # Since ghosts are processed one by one for eaten_ghosts count in the original, 
        # but the order doesn't strictly matter for a single frame (only 1 ghost can be at a position usually, 
        # but even if multiple are, ALE handles them).
        # To match the original logic of incrementing eaten_ghosts, we can use a loop or a clever prefix sum.
        # But there are only 4 ghosts, so let's just use a loop for the eaten_ghosts count if we want to be exact,
        # OR just vmap and sum the rewards.
        
        # Vectorized collision check
        collisions = jax.vmap(detect_collision, in_axes=(None, 0, None))(new_pacman_pos, ghost_positions, consts.COLLISION_THRESHOLD)
        is_frightened = (ghost_modes == GhostMode.FRIGHTENED) | (ghost_modes == GhostMode.BLINKING)
        
        ghosts_eaten = collisions & is_frightened
        deadly_collision = jnp.any(collisions & (~is_frightened))
        
        # Calculate rewards based on how many ghosts were already eaten this power pellet
        # and how many are eaten THIS step.
        base_eaten = jax.lax.cond(ate_power_pellet, lambda: jnp.array(0, dtype=jnp.uint8), lambda: eaten_ghosts)
        
        # Optimized reward calculation using prefix sum (cumsum)
        # Match ALE logic where eating multiple ghosts in one step increases reward multiplier per ghost.
        eaten_this_step = ghosts_eaten.astype(jnp.int32)
        eaten_so_far = jnp.cumsum(eaten_this_step)
        
        # Multiplier for each ghost: base_eaten + its order in this step's eaten batch
        multipliers = base_eaten + eaten_so_far - 1
        rewards = jnp.where(ghosts_eaten, consts.EAT_GHOSTS_BASE_POINTS * (2 ** multipliers.astype(jnp.uint32)), 0)
        total_reward = jnp.sum(rewards)
        final_eaten_count = base_eaten + jnp.sum(eaten_this_step)

        # Update ghost states
        new_ghost_positions = jnp.where(ghosts_eaten[:, None], consts.JAIL_POSITION, ghost_positions)
        new_ghost_actions = jnp.where(ghosts_eaten, Action.NOOP, ghost_actions)
        new_ghost_modes = jnp.where(ghosts_eaten, GhostMode.ENJAILED.value, ghost_modes)
        new_ghost_timers = jnp.where(ghosts_eaten, consts.ENJAILED_DURATION, ghost_timers)

        new_lives = (lives - jnp.where(deadly_collision, 1, 0)).astype(jnp.int8)
        new_death_timer = jnp.where(deadly_collision, consts.RESET_TIMER, 0).astype(jnp.uint32)
        
        return (
            new_ghost_positions,
            new_ghost_actions,
            new_ghost_modes,
            new_ghost_timers,
            final_eaten_count.astype(jnp.uint8),
            new_lives,
            new_death_timer,
            total_reward
        )

    @staticmethod
    def fruit_move(state: PacmanState, dofmaze: chex.Array, key: chex.Array, consts: MsPacmanConstants
                   ) -> Tuple[chex.Array, chex.Array]:
        """
        Updates the fruits position, action and timer if one is currently active.
        """
        # Choose new direction based on last position, action and fruit timer
        allowed = get_allowed_directions(state.fruit.position, state.fruit.action, dofmaze, consts.DIRECTIONS)
        n_allowed = jnp.sum(allowed != 0)
        new_dir = jax.lax.cond(
            n_allowed == 0,
            lambda: state.fruit.action,
            lambda: jax.lax.cond(
                n_allowed == 1,
                lambda: allowed[0],
                lambda: jax.lax.cond(
                    state.fruit.timer == 0,
                    lambda: pathfind(state.fruit.position, state.fruit.action, state.fruit.exit, allowed, key, consts.ACTIONS, consts.DIRECTIONS),
                    lambda: allowed[jax.random.randint(key, (), minval=0, maxval=n_allowed)]
                )
            )
        )
        # Compute the new position
        new_pos = get_new_position(state.fruit.position, new_dir, consts)
        # Return new position and direction values
        return (
            jnp.array(new_pos, dtype=jnp.uint8),
            jnp.array(new_dir, dtype=jnp.uint8)
        )

    @staticmethod
    def fruit_step(state: PacmanState, new_pacman_pos: chex.Array, collected_pellets: chex.Array, dofmaze: chex.Array, key: chex.Array, consts: MsPacmanConstants):
        """
        Updates the fruit state if a fruit spawns, moves or is consumed.
        """
        def spawn_fruit(fruit_spawn: bool):
            fruit_position, fruit_action = get_random_tunnel(state.level.id, key)
            fruit_exit, _ = get_random_tunnel(state.level.id, key)
            return FruitState(
                jnp.array(fruit_position, dtype=jnp.uint8),
                jnp.array(fruit_exit, dtype=jnp.uint8),
                state.fruit.type,
                jnp.array(fruit_action, dtype=jnp.uint8),
                jnp.array(False, dtype=jnp.bool),
                jnp.array(True, dtype=jnp.bool),
                jnp.array(consts.FRUIT_WANDER_DURATION, dtype=jnp.uint16)
            ), 0
        
        def do_nothing(fruit_spawn: bool):
            return FruitState(
                state.fruit.position,
                state.fruit.exit,
                state.fruit.type,
                state.fruit.action,
                state.fruit.spawn | fruit_spawn,
                state.fruit.spawned,
                state.fruit.timer
            ), 0
        
        def consume_fruit(fruit_spawn: bool):
            return FruitState(
                jnp.zeros(2, dtype=jnp.uint8),
                jnp.zeros(2, dtype=jnp.uint8),
                state.fruit.type,
                jnp.array(Action.NOOP, dtype=jnp.uint8),
                state.fruit.spawn | fruit_spawn,
                jnp.array(False, dtype=jnp.bool),
                jnp.array(consts.FRUIT_WANDER_DURATION, dtype=jnp.uint16)
            ), consts.FRUIT_REWARDS[state.fruit.type]
        
        def remove_fruit(fruit_spawn: bool):
            return FruitState(
                jnp.zeros(2, dtype=jnp.uint8),
                jnp.zeros(2, dtype=jnp.uint8),
                state.fruit.type,
                jnp.array(Action.NOOP, dtype=jnp.uint8),
                state.fruit.spawn | fruit_spawn,
                jnp.array(False, dtype=jnp.bool),
                jnp.array(consts.FRUIT_WANDER_DURATION, dtype=jnp.uint16)
            ), 0
        
        def step_fruit(fruit_spawn: bool):
            fruit_type = get_level_fruit(state.level.id, key)
            fruit_position, fruit_action = jax.lax.cond(
                state.step_count % 2 == 0,
                lambda: JaxPacman.fruit_move(state, dofmaze, key, consts),
                lambda: (state.fruit.position, state.fruit.action)
            )
            fruit_timer = jax.lax.cond(
                state.fruit.timer > 0,
                lambda: state.fruit.timer - 1,
                lambda: state.fruit.timer
            )
            return FruitState(
                fruit_position,
                state.fruit.exit,
                jnp.array(fruit_type, dtype=jnp.uint8),
                fruit_action,
                state.fruit.spawn | fruit_spawn,
                state.fruit.spawned,
                fruit_timer
            ), 0
        
        fruit_spawn = jnp.any(consts.FRUIT_SPAWN_THRESHOLDS == collected_pellets) & state.player.has_pellet
        new_fruit_state, reward = jax.lax.cond(
            state.fruit.spawned,
            lambda: jax.lax.cond(
                detect_collision(new_pacman_pos, state.fruit.position, consts.COLLISION_THRESHOLD),
                lambda: consume_fruit(fruit_spawn),
                lambda: jax.lax.cond(
                    (state.fruit.timer == 0) & (jnp.all(jnp.array(state.fruit.position) == jnp.array(state.fruit.exit))),
                    lambda: remove_fruit(fruit_spawn),
                    lambda: step_fruit(fruit_spawn)
                )   
            ),
            lambda: jax.lax.cond(
                state.fruit.spawn,
                lambda: spawn_fruit(fruit_spawn),
                lambda: do_nothing(fruit_spawn)
            )
        ) 
        return new_fruit_state, reward

    @staticmethod
    def flag_score_change(current_score: chex.Array, new_score: chex.Array, consts: MsPacmanConstants):
        """
        Flags the score digits for rendering that changed during the current step.
        """
        powers = 10 ** jnp.arange(consts.MAX_SCORE_DIGITS)
        # Check if (current_score // powers) % 10 != (new_score // powers) % 10
        changed = (current_score // powers) % 10 != (new_score // powers) % 10
        return jnp.flip(changed) # Match digit order: [10^5, 10^4, ..., 10^0]


# -------- Render class --------
class MsPacmanRenderer(JAXGameRenderer):
    """JAX-based MsPacman game renderer, optimized with JIT compilation."""

    def _ensure_palette_color(self, color_rgb: tuple[int, int, int]) -> None:
        color_rgba = (*color_rgb, 255)
        if color_rgb in self.COLOR_TO_ID:
            return
        if color_rgba in self.COLOR_TO_ID:
            self.COLOR_TO_ID[color_rgb] = self.COLOR_TO_ID[color_rgba]
            return
        self.PALETTE, color_id = self.jr.add_palette_color(self.PALETTE, color_rgb)
        self.COLOR_TO_ID[color_rgb] = int(color_id)
        self.COLOR_TO_ID[color_rgba] = int(color_id)

    def _resolve_color_id(self, color_rgb: tuple[int, int, int]) -> int:
        self._ensure_palette_color(color_rgb)
        return self.COLOR_TO_ID.get(color_rgb, self.jr.TRANSPARENT_ID)

    def _build_pacman_oriented_group(self, sprite_path: str) -> list[jnp.ndarray]:
        """
        Build Pacman animation sprites as a single orientation-major group:
        [UP frames][RIGHT frames][LEFT frames][DOWN frames].
        """
        left_frames = [
            self.jr.loadFrame(os.path.join(sprite_path, f"pacman_{i}.npy"))
            for i in range(4)
        ]
        right_frames = [jnp.flip(frame, axis=1) for frame in left_frames]
        up_frames = [jnp.rot90(frame, k=1, axes=(0, 1)) for frame in right_frames]
        down_frames = [jnp.rot90(frame, k=3, axes=(0, 1)) for frame in right_frames]
        return up_frames + right_frames + left_frames + down_frames

    def __init__(self, consts: MsPacmanConstants = None, config: render_utils.RendererConfig = None, sprite_dir_name: str = "mspacman"):
        super().__init__(consts)
        self.consts = consts or MsPacmanConstants()
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), sprite_dir_name)
        
        # Define asset config
        asset_config = [
            {'name': 'dummy_bg', 'type': 'background', 'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)},
            {'name': 'pacman_oriented', 'type': 'group', 'data': self._build_pacman_oriented_group(sprite_path)},
            {'name': 'ghosts', 'type': 'group', 'files': [
                'ghost_blinky.npy', 'ghost_pinky.npy', 'ghost_inky.npy', 'ghost_sue.npy', 
                'ghost_blue.npy', 'ghost_white.npy'
            ]},
            {'name': 'fruit', 'type': 'group', 'files': [
                'fruit_cherry.npy', 'fruit_strawberry.npy', 'fruit_orange.npy',
                'fruit_pretzel.npy', 'fruit_apple.npy', 'fruit_pear.npy', 'fruit_banana.npy'
            ]},
            {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        ]
        
        # Include background colors in the palette (Path, Wall, and Black for UI padding)
        bg_colors = jnp.stack([self.consts.PATH_COLOR, self.consts.WALL_COLOR, jnp.array([0, 0, 0], dtype=jnp.uint8)])
        bg_colors = jnp.concatenate([bg_colors, jnp.full((3, 1), 255, dtype=jnp.uint8)], axis=1)
        asset_config.append({'name': 'bg_colors', 'type': 'procedural', 'data': bg_colors[:, None, :]})

        (self.PALETTE, self.SHAPE_MASKS, _, self.COLOR_TO_ID, self.FLIP_OFFSETS) = \
            self.jr.load_and_setup_assets(asset_config, sprite_path)

        for color in (
            tuple(map(int, self.consts.PATH_COLOR.tolist())),
            tuple(map(int, self.consts.WALL_COLOR.tolist())),
            (0, 0, 0),
        ):
            self._ensure_palette_color(color)

        # Pacman mask group is loaded orientation-major:
        # 0: UP, 1: RIGHT, 2: LEFT, 3: DOWN, each with 4 animation frames.
        pacman_group = self.SHAPE_MASKS['pacman_oriented']
        self.PACMAN_MASKS = pacman_group.reshape(4, 4, pacman_group.shape[1], pacman_group.shape[2])
        
        # Pre-calculate backgrounds for all 4 mazes
        self.MAZE_BACKGROUNDS = self._create_all_backgrounds()

    def _create_all_backgrounds(self):
        bgs = []
        for i in range(4):
            bg = MsPacmanMaze.load_background(i) # Returns (W, H, 3)
            bg = jnp.transpose(bg, (1, 0, 2)) # Convert to (H, W, 3)
            if bg.shape[2] == 3:
                bg = jnp.concatenate([bg, jnp.full((*bg.shape[:2], 1), 255, dtype=jnp.uint8)], axis=2)
            
            bg_id = self.jr._create_background_raster(bg, self.COLOR_TO_ID)
            bgs.append(bg_id)
        return jnp.stack(bgs)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PacmanState):
        maze_idx = get_level_maze(state.level.id)
        background = self.MAZE_BACKGROUNDS[maze_idx]
        raster = self.jr.create_object_raster(background)
        
        # 1. Render Pellets
        wall_id = self._resolve_color_id(tuple(map(int, self.consts.WALL_COLOR.tolist())))
        raster = self.render_pellets(raster, state.level.pellets, wall_id)
        
        # 2. Power Pellets
        raster = self.render_power_pellets(raster, state, wall_id)
        
        # 3. Pacman
        orientation = act_to_dir(state.player.action)
        orientation = jnp.where(orientation == -1, 2, orientation) # Default to LEFT
        frame = (state.step_count & 0b1000) >> 2
        pacman_mask = self.PACMAN_MASKS[orientation.astype(jnp.int32), frame.astype(jnp.int32)]
        raster = self.jr.render_at(raster, state.player.position[0].astype(jnp.int32), state.player.position[1].astype(jnp.int32) - 1, pacman_mask)
        
        # 4. Ghosts
        raster = self.render_ghosts(raster, state)
        
        # 5. Fruit
        raster = jax.lax.cond(
            state.fruit.spawned,
            lambda r: self.jr.render_at(r, state.fruit.position[0].astype(jnp.int32), state.fruit.position[1].astype(jnp.int32) - 1, self.SHAPE_MASKS['fruit'][state.fruit.type.astype(jnp.int32)]),
            lambda r: r,
            raster
        )
        
        # 6. UI
        raster = self.render_ui(raster, state)
        
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def render_pellets(self, raster, pellets, color_id):
        xx, yy = self.jr._xx, self.jr._yy
        
        # Inverse mapping: for each pixel, find which pellet grid cell it belongs to
        # py = gy * 12 + 9, height = 2
        gy = (yy - 9) // 12
        is_pellet_y = ((yy - 9) % 12 >= 0) & ((yy - 9) % 12 < 2)
        
        # px = gx * 8 + (8 if gx < 9 else 12), width = 4
        # The split point is around x = 75-80
        gx = jnp.where(xx < 76, (xx - 8) // 8, (xx - 12) // 8)
        is_pellet_x = jnp.where(xx < 76, 
                                ((xx - 8) % 8 >= 0) & ((xx - 8) % 8 < 4), 
                                ((xx - 12) % 8 >= 0) & ((xx - 12) % 8 < 4))
        
        # Bounds check
        in_grid = (gx >= 0) & (gx < 18) & (gy >= 0) & (gy < 14)
        
        # Correct for jnp.where indexing out of bounds by clipping gx, gy for the lookup
        gx_clipped = jnp.clip(gx, 0, 17)
        gy_clipped = jnp.clip(gy, 0, 13)
        
        # Pellet exists at this grid cell?
        pellet_exists = pellets[gx_clipped, gy_clipped]
        
        should_draw = is_pellet_pixel = in_grid & is_pellet_x & is_pellet_y & pellet_exists
        
        return jnp.where(should_draw, jnp.asarray(color_id, raster.dtype), raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_power_pellets(self, raster, state, color_id):
        # 4x7 sprite
        sprite = jnp.full((7, 4), color_id, dtype=raster.dtype)
        
        should_draw = state.level.power_pellets & (((state.step_count & 0b1000) >> 3) == 1)
        x_coords = (self.consts.POWER_PELLET_TILES[:, 0] * 4 + 4).astype(jnp.int32)
        y_coords = (self.consts.POWER_PELLET_TILES[:, 1] * 4 + 6).astype(jnp.int32)
        
        # Filter positions
        x_coords = jnp.where(should_draw, x_coords, -1)
        y_coords = jnp.where(should_draw, y_coords, -1)
        
        masks = jnp.tile(sprite[None, ...], (4, 1, 1))
        
        return self.jr.render_at_batch(raster, x_coords, y_coords, masks)

    @partial(jax.jit, static_argnums=(0,))
    def render_ghosts(self, raster, state):
        anim_frame = (state.step_count & 0b10000) >> 4
        
        # Determine ghost indices
        is_frightened = (state.ghosts.modes == GhostMode.FRIGHTENED) | (state.ghosts.modes == GhostMode.BLINKING)
        is_blinking_frame = (state.ghosts.modes == GhostMode.BLINKING) & (((state.step_count & 0b1000) >> 3) == 1)
        
        ghost_indices = jnp.where(
            is_frightened,
            jnp.where(is_blinking_frame, 5, 4),
            jnp.arange(4)
        ).astype(jnp.int32)
        
        masks = self.SHAPE_MASKS['ghosts'][ghost_indices]
        
        # Flip masks if needed
        def flip_masks(m):
            return jnp.flip(m, axis=2)
        
        masks = jax.lax.cond(anim_frame == 1, flip_masks, lambda m: m, masks)
        
        x_coords = state.ghosts.positions[:, 0].astype(jnp.int32)
        y_coords = (state.ghosts.positions[:, 1] - 1).astype(jnp.int32)
        
        return self.jr.render_at_batch(raster, x_coords, y_coords, masks)

    @partial(jax.jit, static_argnums=(0,))
    def render_ui(self, raster, state):
        # Score
        digits = self.jr.int_to_digits(state.score, max_digits=self.consts.MAX_SCORE_DIGITS)
        digit_count = get_digit_count(state.score).astype(jnp.int32)
        start_index = self.consts.MAX_SCORE_DIGITS - digit_count
        render_x = 60 + start_index * 8
        raster = self.jr.render_label_selective(raster, render_x, 190, digits, self.SHAPE_MASKS['digits'], start_index, digit_count, spacing=8, max_digits_to_render=self.consts.MAX_SCORE_DIGITS)
        
        # Lives
        life_mask = self.PACMAN_MASKS[1, 1] # Right looking, frame 1
        raster = self.jr.render_indicator(raster, 12, 182, (state.lives - 1).astype(jnp.int32), life_mask, spacing=14, max_value=self.consts.MAX_LIVE_COUNT)
        
        # Fruit indicator
        fruit_mask = self.SHAPE_MASKS['fruit'][state.fruit.type.astype(jnp.int32)]
        raster = self.jr.render_at(raster, 128, 182, fruit_mask)
        
        return raster


# -------- Helper functions --------
def get_digit_count(number: chex.Array):
    """Returns the number of digits in a given decimal number."""
    number = jnp.abs(number)
    return jax.lax.cond(
        number == 0,
        lambda: jnp.array(1, dtype=jnp.uint8),
        lambda: jnp.floor(jnp.log10(number) + 1).astype(jnp.uint8)
    )


def act_to_dir(action: chex.Array):
    """Converts a JAXAtari action into the corresponding DIRECTION index.
    If conversion is not possible -1 is returned.

    action:     2 (UP)  3 (RIGHT)   4 (LEFT)    5 (DOWN)    ELSE
    direction:  0       1           2           3           -1
    """
    return jax.lax.cond(
        (action >= 2) & (action < 6),
        lambda: jnp.array(action - 2, dtype=jnp.int8),
        lambda: jnp.array(-1, dtype=jnp.int8)
    )


def dir_to_act(direction: chex.Array):
    """Converts a DIRECTION index into the corresponding JAXAtari action.
    If conversion is not possible -1 is returned.

    direction:  0       1           2           3           ELSE
    action:     2 (UP)  3 (RIGHT)   4 (LEFT)    5 (DOWN)    -1
    """
    return jax.lax.cond(
        (direction >= 0) & (direction < 4),
        lambda: jnp.array(direction + 2, dtype=jnp.int8),
        lambda: jnp.array(-1, dtype=jnp.int8)
    )


def last_pressed_action(action, prev_action):
    """Returns the last pressed action in cases where both actions are pressed"""
    return jax.lax.cond(
        action == Action.UPRIGHT,
        lambda: jax.lax.cond(
            prev_action == Action.UP,
            lambda: Action.RIGHT,
            lambda: Action.UP
        ),
        lambda: jax.lax.cond(
            action == Action.UPLEFT,
            lambda: jax.lax.cond(
                prev_action == Action.UP,
                lambda: Action.LEFT,
                lambda: Action.UP
            ),
            lambda: jax.lax.cond(
                action == Action.DOWNRIGHT,
                lambda: jax.lax.cond(
                    prev_action == Action.DOWN,
                    lambda: Action.RIGHT,
                    lambda: Action.DOWN
                ),
                lambda: jax.lax.cond(
                    action == Action.DOWNLEFT,
                    lambda: jax.lax.cond(
                        prev_action == Action.DOWN,
                        lambda: Action.LEFT,
                        lambda: Action.DOWN
                    ),
                    lambda: action
                )
            )
        )
    )


def dof(pos: chex.Array, dofmaze: chex.Array):
    """Degree of freedom of the object, can it move up, right, left, down"""
    x, y = pos
    grid_x = (x + 5) // 4
    grid_y = (y + 3) // 4
    return dofmaze[grid_x, grid_y]


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
    return jnp.array([
        up & on_vertical_grid,
        right & on_horizontal_grid,
        left & on_horizontal_grid,
        down & on_vertical_grid
    ], dtype=jnp.bool_)


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
    return jnp.array([
        ~up & on_horizontal_grid,
        ~right & on_vertical_grid,
        ~left & on_vertical_grid,
        ~down & on_horizontal_grid
    ], dtype=jnp.bool_)


def get_allowed_directions(position: chex.Array, action: chex.Array, dofmaze: chex.Array, directions: chex.Array, is_ghost: bool = False):
    """
    Returns an array of all directions (JAXAtari actions) in which movement is possible.
    To be jit-compatible the size of the output array is fixed, so invalid directions are marked with 0 (NOOP).
    Turning is only allowed at the centre of each tile and reverting is not allowed.
    """
    direction_count = directions.shape[0]

    # Return allowed directions and their count
    def at_center(_):
        # Available directions for the current position
        available_mask = available_directions(position, dofmaze)
        
        # Restrict ghosts from entering side tunnels
        available_mask = jax.lax.cond(
            jnp.array(is_ghost, dtype=jnp.bool_),
            lambda: available_mask.at[1].set(jnp.where(position[0] >= 132, False, available_mask[1])).at[2].set(jnp.where(position[0] <= 28, False, available_mask[2])),
            lambda: available_mask
        )

        # Directions that are not the reverse of current action
        not_reverse_mask = jnp.arange(direction_count) != act_to_dir(reverse_action(action))

        allowed_mask = available_mask & not_reverse_mask
        allowed_actions = jnp.where(allowed_mask, directions, 0)
        return jnp.compress(allowed_actions != 0, allowed_actions, size=direction_count).astype(jnp.uint8)

    # Return the current direction
    def not_at_center(_):
        return jnp.zeros(direction_count, dtype=jnp.uint8).at[0].set(action)

    # Check if the position is at the center of a tile
    at_tile_center = (position[0] % 4 == 1) | (position[1] % 12 == 6)
    return jax.lax.cond(
        at_tile_center,
        at_center,
        not_at_center,
        None
    )


def get_chase_target(ghost: GhostType,
                     ghost_position: chex.Array, blinky_pos: chex.Array,
                     player_pos: chex.Array, player_dir: chex.Array,
                     actions: chex.Array, scatter_targets: chex.Array) -> chex.Array:
    """
    Compute the chase-mode target for each ghost:
    0=Red (Blinky), 1=Pink (Pinky), 2=Blue (Inky), 3=Orange (Sue)
    """
    def get_blinky_target(_):
        return player_pos
    
    def get_pinky_target(_):
        return player_pos + 4*MsPacmanMaze.TILE_SCALE * actions[player_dir]
    
    def get_inky_target(_):
        two_ahead = player_pos + 2*MsPacmanMaze.TILE_SCALE * actions[player_dir]
        vect = two_ahead - blinky_pos
        return blinky_pos + 2 * vect
    
    def get_sue_target(_):
        dist = jnp.linalg.norm(ghost_position - player_pos)
        return jnp.where(dist > 8*MsPacmanMaze.TILE_SCALE, player_pos, scatter_targets[GhostType.SUE])
    
    return jax.lax.switch(
        ghost,
        (
            get_blinky_target,  # GhostType.BLINKY
            get_pinky_target,   # GhostType.PINKY
            get_inky_target,    # GhostType.INKY
            get_sue_target      # GhostType.SUE
        ),
        None
    )


def pathfind(position: chex.Array, direction: chex.Array, target: chex.Array, allowed: chex.Array, key: chex.Array, actions: chex.Array, directions: chex.Array):
    """
    Returns the direction which should be taken to approach the target.
    If multiple options exist the direction is chosen that minimizes the distance on the longer axis - horizontal or vertical.
    If both distances are equal or multiple options exist on the same axis, the direction is chosen randomly.
    """
    valid_mask = allowed != 0
    n_allowed = jnp.sum(valid_mask)

    # If no direction allowed - Continue forward
    def no_allowed():
        return direction.astype(allowed.dtype)

    # If one direction allowed - Take it
    def one_allowed():
        return allowed[0].astype(allowed.dtype)

    # If multiple directions allowed - Get cost of all possible steps and determine advantageous directions
    def multi_allowed():
        new_positions = position + actions[allowed]
        costs = jnp.abs(new_positions - target).sum(axis=1)  # Manhattan distances
        costs = jnp.where(valid_mask, costs, jnp.iinfo(jnp.int32).max)
        min_cost = jnp.min(costs)
        min_mask = costs == min_cost
        min_dirs = jnp.compress(min_mask, allowed, size=directions.shape[0])
        n_min = jnp.sum(min_dirs != 0)

        # If one direction advantageous - Take it
        def one_min():
            return min_dirs[0].astype(allowed.dtype)

        # If multiple directions advantageous - Prioritize the longer axis
        def multi_min():
            h_dist = jnp.abs(position[0] - target[0])
            v_dist = jnp.abs(position[1] - target[1])
            h_dirs = jnp.array([int(Action.LEFT), int(Action.RIGHT)], dtype=jnp.int32)
            v_dirs = jnp.array([int(Action.DOWN), int(Action.UP)], dtype=jnp.int32)
            h_mask = jnp.isin(min_dirs, h_dirs)
            v_mask = jnp.isin(min_dirs, v_dirs)
            prefer_h = h_dist >= v_dist
            prefer_v = v_dist >= h_dist
            prefered = (h_mask & prefer_h) | (v_mask & prefer_v)
            n_prefered = jnp.sum(prefered)

            # If no direction advantageous on longer axis - Choose randomly
            def no_long_axis():
                return min_dirs[jax.random.randint(key, (), 0, n_min)].astype(allowed.dtype)

            # If one direction advantageous on longer axis - Take it
            def one_long_axis():
                return min_dirs[jnp.argmax(prefered)].astype(allowed.dtype)
            
            # If multiple directions advantageous on longer or equal axis - Choose randomly with mask
            def multi_long_axis():
                prefered_dirs = jnp.compress(prefered, min_dirs, size=directions.shape[0])
                return prefered_dirs[jax.random.randint(key, (), 0, n_prefered)].astype(allowed.dtype)

            # Check for advantageous directions on longer axis
            return jax.lax.cond(
                n_prefered == 0,
                no_long_axis,
                lambda: jax.lax.cond(
                    n_prefered == 1,
                    one_long_axis,
                    multi_long_axis
                )
            )

        # Check for advantageous directions
        return jax.lax.cond(
            n_min == 1,
            one_min,
            multi_min
        )

    # Check for allowed directions
    return jax.lax.cond(
        n_allowed == 0,
        no_allowed,
        lambda: jax.lax.cond(
            n_allowed == 1,
            one_allowed,
            multi_allowed
        )
    )


"""Returns the next position, given the current position and action that is applied this step"""
def get_new_position(position: chex.Array, action: chex.Array, consts: MsPacmanConstants):
    new_position = position + consts.ACTIONS[action]
    return new_position.at[0].set(new_position[0] % 160)  # Wrap around horizontally for tunnels



def get_level_maze(level: chex.Array):
    """Returns the maze id that correpsonds to the current level."""
    return jax.lax.switch(  # Invalid levels (<0) are not handled explicitly and just get assigned to level 0
        jnp.digitize(level, jnp.array([2, 4, 6, 8]), right=True).astype(jnp.int32),
        (
            lambda lvl: jnp.array(0, dtype=jnp.int32),  # Levels 1-2
            lambda lvl: jnp.array(1, dtype=jnp.int32),  # Levels 3-4
            lambda lvl: jnp.array(2, dtype=jnp.int32),  # Levels 5-6
            lambda lvl: jnp.array(3, dtype=jnp.int32),  # Levels 7-8
            lambda lvl: jax.lax.cond(                   # Levels 9+
                (lvl % 4 == 0) | (lvl % 4 == 1),
                lambda: jnp.array(2, dtype=jnp.int32),
                lambda: jnp.array(3, dtype=jnp.int32)
            )
        ),
        level
    )


def get_level_fruit(level: chex, key: chex.Array):
    """Returns the fruit that corresponds to the current level."""
    return jax.lax.cond(
        level > 7,
        lambda: jax.random.randint(key, (), 0, 6),
        lambda: jax.lax.switch(
            level,
            (
                lambda _: FruitType.CHERRY,     # Level 0 - Invalid level, defaults to cherry
                lambda _: FruitType.CHERRY,     # Level 1
                lambda _: FruitType.STRAWBERRY, # Level 2
                lambda _: FruitType.ORANGE,     # Level 3
                lambda _: FruitType.PRETZEL,    # Level 4
                lambda _: FruitType.APPLE,      # Level 5
                lambda _: FruitType.PEAR,       # Level 6
                lambda _: FruitType.BANANA      # Level 7
            ),
            None
        )
    )


def get_random_tunnel(level: chex.Array, key: chex.Array):
    """Returns the position and exit direction of a random tunnel."""
    maze = get_level_maze(level)
    tunnel_heights = jnp.array(MsPacmanMaze.TUNNEL_HEIGHTS)[maze]

    tunnels_dir = jnp.array([
        int(Action.RIGHT),
        int(Action.LEFT),
        int(Action.RIGHT),
        int(Action.LEFT)
    ], dtype=jnp.uint8)

    tunnels_pos = jnp.array([
        [0, tunnel_heights[0]],
        [MsPacmanMaze.WIDTH - 1, tunnel_heights[0]],
        [0, tunnel_heights[1]],
        [MsPacmanMaze.WIDTH - 1, tunnel_heights[1]]
    ], dtype=jnp.int32)

    # If the second element is 0, there is only one pair of tunnels
    max_choices = jax.lax.cond(tunnel_heights[1] == 0, lambda: 2, lambda: 4)
    tunnel_idx = jax.random.randint(key, (), 0, max_choices)

    return tunnels_pos[tunnel_idx], tunnels_dir[tunnel_idx]


def reverse_action(dir_idx: chex.Array):
    """Inverts the direction if possible."""
    # Mapping for actions: 0->0, 1->1, 2->5, 3->4, 4->3, 5->2
    inv_map = jnp.array([0, 1, 5, 4, 3, 2], dtype=jnp.uint8)
    idx = jnp.array(dir_idx, dtype=jnp.uint8)
    in_range = (idx >= 0) & (idx < inv_map.shape[0])
    return jnp.where(in_range, inv_map[idx], idx).astype(idx.dtype)
    

def detect_collision(position_1: chex.Array, position_2: chex.Array, collision_threshold: chex.Array):
    """Checks if the two positions are closer than the collision threshold."""
    return jnp.all(abs(jnp.array(position_1) - jnp.array(position_2)) < collision_threshold)


# -------- Reset functions --------
def reset_level(level: chex.Array):
    return LevelState(
        id                  = jnp.array(level, dtype=jnp.uint8),
        collected_pellets   = jnp.array(0, dtype=jnp.uint8),
        pellets             = jnp.copy(MsPacmanMaze.BASE_PELLETS),
        power_pellets       = jnp.ones(4, dtype=jnp.bool_),
        loaded              = jnp.array(0, dtype=jnp.uint8)
    )

def reset_player(consts: MsPacmanConstants):
    return PlayerState(
        position            = consts.INITIAL_PACMAN_POSITION,
        action              = jnp.array(Action.LEFT, dtype=jnp.uint8),
        has_pellet          = jnp.array(False),
        eaten_ghosts        = jnp.array(0, dtype=jnp.uint8)
    )

def reset_ghosts(consts: MsPacmanConstants):
    return GhostsState (
        positions   = consts.INITIAL_GHOSTS_POSITIONS,
        types       = jnp.array([GhostType.BLINKY, GhostType.PINKY, GhostType.INKY, GhostType.SUE], dtype=jnp.uint8),
        actions     = jnp.array([Action.LEFT, Action.NOOP, Action.NOOP, Action.NOOP], dtype=jnp.uint8),
        modes       = jnp.array([GhostMode.RANDOM, GhostMode.ENJAILED, GhostMode.ENJAILED, GhostMode.ENJAILED], dtype=jnp.uint8),
        timers      = jnp.array([consts.SCATTER_DURATION, consts.PINKY_RELEASE_TIME, consts.INKY_RELEASE_TIME, consts.SUE_RELEASE_TIME], dtype=jnp.float16),
    )

def reset_fruit(consts: MsPacmanConstants, level: chex.Array, key: chex.PRNGKey):
    return FruitState(
        position    = jnp.zeros(2, dtype=jnp.uint8),
        exit        = jnp.zeros(2, dtype=jnp.uint8),
        type        = jnp.array(get_level_fruit(level, key), dtype=jnp.uint8),
        action      = jnp.array(Action.NOOP, dtype=jnp.uint8),
        spawn       = jnp.array(False, dtype=jnp.bool),
        spawned     = jnp.array(False, dtype=jnp.bool),
        timer       = jnp.array(jnp.array(consts.FRUIT_WANDER_DURATION, dtype=jnp.uint16), dtype=jnp.uint16)
    )

def reset_game(consts: MsPacmanConstants, level: chex.Array, lives: chex.Array, score: chex.Array, key: chex.PRNGKey):
    return PacmanState(
        level           = reset_level(level),
        player          = reset_player(consts),
        ghosts          = reset_ghosts(consts),
        fruit           = reset_fruit(consts, level, key),
        lives           = jnp.array(lives, dtype=jnp.int8),
        score           = jnp.array(score, dtype=jnp.uint32),
        score_changed   = jnp.arange(consts.MAX_SCORE_DIGITS) >= (consts.MAX_SCORE_DIGITS - get_digit_count(score)),
        freeze_timer    = jnp.array(0, dtype=jnp.uint32),
        step_count      = jnp.array(0, dtype=jnp.uint32),
        key             = key
    )

def reset_entities(consts: MsPacmanConstants, state: PacmanState, key: chex.PRNGKey):
    return PacmanState(
        level = LevelState(
            id = state.level.id,
            collected_pellets=state.level.collected_pellets,
            pellets=state.level.pellets,
            power_pellets=state.level.power_pellets,
            loaded=state.level.loaded
        ),
        player          = reset_player(consts),
        ghosts          = reset_ghosts(consts),
        fruit           = reset_fruit(consts, state.level.id, key),
        lives           = state.lives,
        score           = state.score,
        score_changed   = state.score_changed,
        freeze_timer    = state.freeze_timer,
        step_count      = jnp.array(0, dtype=jnp.uint32),
        key             = state.key
    )
