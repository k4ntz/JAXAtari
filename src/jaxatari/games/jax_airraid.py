import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, List, Union
import jax
import jax.numpy as jnp
import jax.random as random
import chex
import pygame
import jaxatari.spaces as spaces

# from jaxatari.rendering import atraJaxis as aj
# from jaxatari.environment import JaxEnvironment, JAXAtariAction
# from jaxatarifrom jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer

# Game environment
WIDTH: int = 160
HEIGHT: int = 210

# Player
PLAYER_WIDTH: int = 14
PLAYER_HEIGHT: int = 12
PLAYER_SPEED: int = 3
PLAYER_INITIAL_X: int = 80
PLAYER_INITIAL_Y: int = 140

# Buildings
NUM_BUILDINGS: int = 2
BUILDING_WIDTH: int = 50
BUILDING_HEIGHT: int = 25
MAX_BUILDING_DAMAGE: int = 6
BUILDING_INITIAL_Y: int = 160
BUILDING_VELOCITY: int = 1
BUILDING_SPACING: int = 90

# Height and Y position based on damage level
BUILDING_HEIGHTS: chex.Array = jnp.array([25, 21, 17, 13, 9, 5, 0])
BUILDING_Y_POSITIONS: chex.Array = jnp.array([160, 164, 168, 172, 176, 180, 190])

# Enemies
NUM_ENEMIES_PER_TYPE: int = 3
TOTAL_ENEMIES: int = 9  # NUM_ENEMIES_PER_TYPE * 4
ENEMY_INITIAL_Y: int = 69
ENEMY_SPEED: float = 1.5
ENEMY_SPAWN_Y: int = 30
ENEMY_SPAWN_PROB: float = 0.02

# Missiles
MISSILE_WIDTH: int = 2
MISSILE_HEIGHT: int = 2
NUM_PLAYER_MISSILES: int = 1
NUM_ENEMY_MISSILES: int = 1
PLAYER_MISSILE_SPEED: int = -6
ENEMY_MISSILE_SPEED: int = 4
ENEMY_FIRE_PROB: float = 0.05

class AirRaidConstants(NamedTuple):
    # Game environment
    WIDTH: int = 160
    HEIGHT: int = 210

    # Player
    PLAYER_WIDTH: int = 14
    PLAYER_HEIGHT: int = 12
    PLAYER_SPEED: int = 3
    PLAYER_INITIAL_X: int = 80
    PLAYER_INITIAL_Y: int = 140

    # Buildings
    NUM_BUILDINGS: int = 2
    BUILDING_WIDTH: int = 50
    BUILDING_HEIGHT: int = 25
    MAX_BUILDING_DAMAGE: int = 6
    BUILDING_INITIAL_Y: int = 160
    BUILDING_VELOCITY: int = 1
    BUILDING_SPACING: int = 90

    # Height and Y position based on damage level
    BUILDING_HEIGHTS: chex.Array = jnp.array([25, 21, 17, 13, 9, 5, 0])
    BUILDING_Y_POSITIONS: chex.Array = jnp.array([160, 164, 168, 172, 176, 180, 190])

    # Enemies
    NUM_ENEMIES_PER_TYPE: int = 3
    TOTAL_ENEMIES: int = 9  # NUM_ENEMIES_PER_TYPE * 4
    ENEMY_INITIAL_Y: int = 69
    ENEMY_SPEED: float = 1.5
    ENEMY_SPAWN_Y: int = 30
    ENEMY_SPAWN_PROB: float = 0.02

    # Missiles
    MISSILE_WIDTH: int = 2
    MISSILE_HEIGHT: int = 2
    NUM_PLAYER_MISSILES: int = 1
    NUM_ENEMY_MISSILES: int = 1
    PLAYER_MISSILE_SPEED: int = -6
    ENEMY_MISSILE_SPEED: int = 4
    ENEMY_FIRE_PROB: float = 0.05

def get_human_action() -> chex.Array:
    """
    Records if LEFT, RIGHT, or FIRE is being pressed and returns the corresponding action.
    Returns: action taken by the player.
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(Action.LEFTFIRE)
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(Action.RIGHTFIRE)
    elif keys[pygame.K_LEFT]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_RIGHT]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    else:
        return jnp.array(Action.NOOP)

# Immutable state container
class AirRaidState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_lives: chex.Array

    building_x: chex.Array
    building_y: chex.Array
    building_damage: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_active: chex.Array
    enemy_has_fired: chex.Array  # Track which enemies have already fired

    player_missile_x: chex.Array
    player_missile_y: chex.Array
    player_missile_active: chex.Array

    enemy_missile_x: chex.Array
    enemy_missile_y: chex.Array
    enemy_missile_active: chex.Array

    score: chex.Array
    step_counter: chex.Array
    flash_counter: chex.Array  # Counter for screen flashing animation (building damage or game over)
    rng: chex.Array  # Random key for stochastic game elements

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class AirRaidObservation(NamedTuple):
    player: EntityPosition
    buildings: jnp.ndarray
    enemies: jnp.ndarray
    player_missiles: jnp.ndarray
    enemy_missiles: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray

class AirRaidInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

@jax.jit
def player_step(player_x: chex.Array, action: chex.Array) -> chex.Array:
    """
    Updates the player position based on the action.

    Args:
        player_x: Current player x position
        action: Action taken by player

    Returns:
        New player x position
    """
    # 20px boundary on each side to prevent hiding at edges
    LEFT_BOUNDARY = 10
    RIGHT_BOUNDARY = WIDTH - PLAYER_WIDTH - 10

    # Check if left or right button was pressed
    move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
    move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

    player_x = jnp.where(
        move_left,
        jnp.maximum(player_x - PLAYER_SPEED, LEFT_BOUNDARY),
        player_x
    )

    player_x = jnp.where(
        move_right,
        jnp.minimum(player_x + PLAYER_SPEED, RIGHT_BOUNDARY),
        player_x
    )

    return player_x

@jax.jit
def spawn_enemy(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Spawns a new enemy if conditions are met and position doesn't overlap with existing enemies.
    Args: state: Current game state
    Returns: Updated enemy arrays including has_fired status
    """
    rng, spawn_key, type_key, pos_key1, pos_key2 = random.split(state.rng, 5)
    spawn_prob = random.uniform(spawn_key)

    # Find the first inactive enemy slot
    inactive_mask = 1 - state.enemy_active
    first_inactive = jnp.max(jnp.where(inactive_mask, jnp.arange(TOTAL_ENEMIES), -1))

    # Generate new enemy properties
    new_type = random.randint(type_key, shape=(), minval=0, maxval=4)
    new_width = jnp.where(new_type == 0, 16, 14)  # Simplified: type 0 = 16px, others = 14px

    # Helper function for overlap checking
    def has_overlap(x):
        active_in_spawn = jnp.logical_and(state.enemy_active == 1, state.enemy_y < 120)
        existing_widths = jnp.where(state.enemy_type == 0, 16, 14)
        return jnp.any(jnp.logical_and(
            active_in_spawn,
            jnp.logical_and(x < state.enemy_x + existing_widths, x + new_width > state.enemy_x)
        ))

    # Try two candidate positions
    candidates = jnp.array([
        random.randint(pos_key1, shape=(), minval=10, maxval=WIDTH - 30),
        random.randint(pos_key2, shape=(), minval=10, maxval=WIDTH - 30)
    ])

    # Check overlaps and select first valid position
    overlaps = jnp.array([has_overlap(candidates[0]), has_overlap(candidates[1])])
    valid_candidates = ~overlaps
    new_x = jnp.where(valid_candidates[0], candidates[0], candidates[1])

    # Spawn conditions: probability + slot available + at least one valid position
    should_spawn = jnp.logical_and(
        jnp.logical_and(spawn_prob < ENEMY_SPAWN_PROB, first_inactive >= 0),
        jnp.any(valid_candidates)
    )

    # Update enemy arrays
    enemy_x = state.enemy_x.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(new_x), state.enemy_x[first_inactive]))
    enemy_y = state.enemy_y.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(ENEMY_SPAWN_Y), state.enemy_y[first_inactive]))
    enemy_type = state.enemy_type.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(new_type), state.enemy_type[first_inactive]))
    enemy_active = state.enemy_active.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(1), state.enemy_active[first_inactive]))
    enemy_has_fired = state.enemy_has_fired.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(0), state.enemy_has_fired[first_inactive]))  # Reset firing status

    return enemy_x, enemy_y, enemy_type, enemy_active, enemy_has_fired, rng

@jax.jit
def update_enemies(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Updates all enemy positions. Enemies move down the screen."""
    # Extract state components
    enemy_y = state.enemy_y
    enemy_active = state.enemy_active
    enemy_has_fired = state.enemy_has_fired
    building_damage = state.building_damage

    # Move active enemies down
    enemy_y = jnp.where(enemy_active == 1, enemy_y + jnp.int32(ENEMY_SPEED), enemy_y)

    # Deactivate enemies that reach the bottom
    reached_player = enemy_y > jnp.int32(PLAYER_INITIAL_Y - 20)  # Changed from HEIGHT to PLAYER_INITIAL_Y
    enemy_active = jnp.where(reached_player, jnp.int32(0), enemy_active)

    # Reset firing status for deactivated enemies
    enemy_has_fired = jnp.where(reached_player, jnp.int32(0), enemy_has_fired)

    return enemy_y, enemy_active, enemy_has_fired, building_damage


@jax.jit
def fire_player_missile(state: AirRaidState, action: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Creates a new player missile if FIRE action is taken and a missile slot is available.

    Args:
        state: Current game state
        action: Player action

    Returns:
        Updated player missile positions and active flags
    """
    # Check if fire button was pressed
    is_fire = jnp.logical_or(
        jnp.logical_or(action == Action.FIRE, action == Action.LEFTFIRE),
        action == Action.RIGHTFIRE
    )

    # Find the first inactive missile
    inactive_missile_mask = 1 - state.player_missile_active
    inactive_indices = jnp.where(inactive_missile_mask, jnp.arange(NUM_PLAYER_MISSILES), -1)
    first_inactive = jnp.max(inactive_indices)

    # Only fire if button pressed and missile slot is available
    should_fire = jnp.logical_and(is_fire, first_inactive >= 0)

    missile_x = state.player_x + (PLAYER_WIDTH // 2) - (MISSILE_WIDTH // 2)


    # Update missile state if firing
    player_missile_x = state.player_missile_x.at[first_inactive].set(
        jnp.where(should_fire, missile_x, state.player_missile_x[first_inactive])
    )
    player_missile_y = state.player_missile_y.at[first_inactive].set(
        jnp.where(should_fire, state.player_y, state.player_missile_y[first_inactive])
    )
    player_missile_active = state.player_missile_active.at[first_inactive].set(
        jnp.where(should_fire, 1, state.player_missile_active[first_inactive])
    )

    return player_missile_x, player_missile_y, player_missile_active

@jax.jit
def fire_enemy_missiles(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Randomly generates enemy missiles from active enemies that haven't fired yet.

    Args:
        state: Current game state

    Returns:
        Updated enemy missile positions, active flags, enemy_has_fired status, and RNG
    """
    rng = state.rng
    enemy_missile_x = state.enemy_missile_x
    enemy_missile_y = state.enemy_missile_y
    enemy_missile_active = state.enemy_missile_active

    # Find the first inactive missile
    inactive_missile_mask = 1 - enemy_missile_active
    inactive_indices = jnp.where(inactive_missile_mask, jnp.arange(NUM_ENEMY_MISSILES), -1)
    first_inactive = jnp.max(inactive_indices)  # Get the highest valid index

    # Generate random values for firing decision and which enemy fires
    rng, fire_key, enemy_key = random.split(rng, 3)
    fire_prob = random.uniform(fire_key)

    # Count active enemies without using nonzero
    active_enemy_count = jnp.sum(state.enemy_active)

    # Randomly select an enemy index (0 to TOTAL_ENEMIES-1)
    random_enemy_idx = random.randint(enemy_key, shape=(), minval=0, maxval=TOTAL_ENEMIES)

    # We'll iterate through the enemies and select the first active one after our random index
    # This is a workaround since we can't use jnp.nonzero in jitted code

    # This function finds a valid active enemy that hasn't fired yet
    def find_active_unfired_enemy(random_idx, enemy_active, enemy_has_fired):
        # Create a shifted array where we start checking from random_idx
        indices = (random_idx + jnp.arange(TOTAL_ENEMIES)) % TOTAL_ENEMIES

        # For each index, check if it's active AND hasn't fired yet
        can_fire = jnp.logical_and(enemy_active[indices] == 1, enemy_has_fired[indices] == 0)

        # Compute scores - unfired active enemies get high scores
        scores = jnp.where(
            can_fire,
            TOTAL_ENEMIES - jnp.arange(TOTAL_ENEMIES),
            -1
        )

        # Find the index with the highest score (first active unfired enemy)
        best_idx = indices[jnp.argmax(scores)]

        # Return the best enemy index, or 0 if none found
        return jnp.where(jnp.max(scores) >= 0, best_idx, 0)

    # Find a valid active enemy that hasn't fired
    firing_enemy_idx = find_active_unfired_enemy(random_enemy_idx, state.enemy_active, state.enemy_has_fired)

    # Only fire if probability is met, enemy is available, there's an inactive missile slot,
    # there are no currently active enemy missiles, AND the selected enemy hasn't fired yet
    enemy_available = active_enemy_count > 0
    enemy_hasnt_fired = state.enemy_has_fired[firing_enemy_idx] == 0
    no_active_missiles = jnp.sum(enemy_missile_active) == 0  # Ensure no missiles are currently active
    can_fire = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(fire_prob < ENEMY_FIRE_PROB, first_inactive >= 0),
            jnp.logical_and(enemy_available, enemy_hasnt_fired)
        ),
        no_active_missiles
    )

    enemy_width = jnp.where(
        state.enemy_type[firing_enemy_idx] == 0, 16,  # Enemy25 width
        jnp.where(state.enemy_type[firing_enemy_idx] < 3, 14, 14)  # Enemy50/75 width, Enemy100 width
    )

    # Update missile state if firing
    enemy_missile_x = enemy_missile_x.at[first_inactive].set(
        jnp.where(
            can_fire,
            state.enemy_x[firing_enemy_idx] + enemy_width // 2,
            enemy_missile_x[first_inactive]
        )
    )

    enemy_missile_y = enemy_missile_y.at[first_inactive].set(
        jnp.where(
            can_fire,
            state.enemy_y[firing_enemy_idx] + (
                jnp.where(state.enemy_type[firing_enemy_idx] == 0, 18,
                      jnp.where(state.enemy_type[firing_enemy_idx] < 3, 16, 14))
            ),
            enemy_missile_y[first_inactive]
        )
    )

    enemy_missile_active = enemy_missile_active.at[first_inactive].set(
        jnp.where(can_fire, 1, enemy_missile_active[first_inactive])
    )

    # Mark the enemy as having fired
    enemy_has_fired = state.enemy_has_fired.at[firing_enemy_idx].set(
        jnp.where(can_fire, 1, state.enemy_has_fired[firing_enemy_idx])
    )

    return enemy_missile_x, enemy_missile_y, enemy_missile_active, enemy_has_fired, rng

@jax.jit
def update_missiles(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Updates the positions of all missiles and deactivates those that go off-screen.

    Args:
        state: Current game state

    Returns:
        Updated player and enemy missile positions and active flags
    """
    # Move player missiles up
    player_missile_y = jnp.where(
        state.player_missile_active == 1,
        state.player_missile_y + PLAYER_MISSILE_SPEED,
        state.player_missile_y
    )

    # Move enemy missiles down
    enemy_missile_y = jnp.where(
        state.enemy_missile_active == 1,
        state.enemy_missile_y + ENEMY_MISSILE_SPEED,
        state.enemy_missile_y
    )

    # Deactivate missiles that go off-screen
    player_missile_active = jnp.where(
        player_missile_y < 0,
        0,
        state.player_missile_active
    )

    enemy_missile_active = jnp.where(
        enemy_missile_y > HEIGHT,
        0,
        state.enemy_missile_active
    )

    return player_missile_y, player_missile_active, enemy_missile_y, enemy_missile_active

@jax.jit
def detect_collisions(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Detects all collisions between game objects."""
    enemy_active = state.enemy_active
    player_missile_active = state.player_missile_active
    enemy_missile_active = state.enemy_missile_active
    score = state.score
    player_lives = state.player_lives
    building_damage = state.building_damage

    # Check player missiles hitting enemies
    for pm in range(NUM_PLAYER_MISSILES):
        is_missile_active = player_missile_active[pm]

        for e in range(TOTAL_ENEMIES):
            is_enemy_active = enemy_active[e]

            # Get enemy dimensions based on type
            enemy_width = jnp.where(
                state.enemy_type[e] == 0, 16,
                jnp.where(state.enemy_type[e] < 3, 14, 14)
            )

            enemy_height = jnp.where(
                state.enemy_type[e] == 0, 18,
                jnp.where(state.enemy_type[e] < 3, 16, 14)
            )

            # Check collision
            collision = jnp.logical_and(
                jnp.logical_and(
                    state.player_missile_x[pm] < state.enemy_x[e] + enemy_width,
                    state.player_missile_x[pm] + MISSILE_WIDTH > state.enemy_x[e]
                ),
                jnp.logical_and(
                    state.player_missile_y[pm] < state.enemy_y[e] + enemy_height,
                    state.player_missile_y[pm] + MISSILE_HEIGHT > state.enemy_y[e]
                )
            )

            # Only count collision if both objects are active
            effective_collision = jnp.logical_and(
                jnp.logical_and(collision, is_missile_active),
                is_enemy_active
            )

            enemy_active = enemy_active.at[e].set(
                jnp.where(effective_collision, 0, enemy_active[e])
            )

            player_missile_active = player_missile_active.at[pm].set(
                jnp.where(effective_collision, 0, player_missile_active[pm])
            )

            score_values = jnp.array([25, 50, 75, 100])
            score_to_add = score_values[state.enemy_type[e]]
            score = jnp.where(effective_collision, score + score_to_add, score)

    # Check enemy missiles hitting buildings and player
    for em in range(NUM_ENEMY_MISSILES):
        is_missile_active = enemy_missile_active[em]

        for b in range(NUM_BUILDINGS):
            collision = jnp.logical_and(
                jnp.logical_and(
                    state.enemy_missile_x[em] >= state.building_x[b],
                    state.enemy_missile_x[em] < state.building_x[b] + BUILDING_WIDTH
                ),
                jnp.logical_and(
                    state.enemy_missile_y[em] >= BUILDING_Y_POSITIONS[building_damage[b]],
                    state.enemy_missile_y[em] < BUILDING_Y_POSITIONS[building_damage[b]] + BUILDING_HEIGHTS[building_damage[b]]
                )
            )

            effective_collision = jnp.logical_and(collision, is_missile_active == 1)

            building_damage = building_damage.at[b].set(
                jnp.where(effective_collision,
                         jnp.minimum(building_damage[b] + 1, MAX_BUILDING_DAMAGE),
                         building_damage[b])
            )
            enemy_missile_active = enemy_missile_active.at[em].set(
                jnp.where(effective_collision, 0, enemy_missile_active[em])
            )

        player_collision = jnp.logical_and(
            jnp.logical_and(
                state.enemy_missile_x[em] < state.player_x + PLAYER_WIDTH,
                state.enemy_missile_x[em] + MISSILE_WIDTH > state.player_x
            ),
            jnp.logical_and(
                state.enemy_missile_y[em] < state.player_y + PLAYER_HEIGHT,
                state.enemy_missile_y[em] + MISSILE_HEIGHT > state.player_y
            )
        )

        effective_player_collision = jnp.logical_and(player_collision, is_missile_active == 1)

        enemy_missile_active = enemy_missile_active.at[em].set(
            jnp.where(effective_player_collision, 0, enemy_missile_active[em])
        )

        player_lives = jnp.where(effective_player_collision, player_lives - 1, player_lives)

    return enemy_active, player_missile_active, enemy_missile_active, score, player_lives, building_damage


class JaxAirRaid(JaxEnvironment[AirRaidState, AirRaidObservation, AirRaidInfo, AirRaidConstants]):
    def __init__(self, consts: AirRaidConstants = None, frameskip: int = 0, reward_funcs: list = None):
        consts = consts or AirRaidConstants()
        super().__init__(consts)
        self.frameskip = frameskip + 1
        if reward_funcs is not None:
            self.reward_funcs = tuple(reward_funcs)
        else:
            self.reward_funcs = None
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE
        }
        self.renderer = AirRaidRenderer(consts)

    def render(self, state: AirRaidState) -> jnp.ndarray:
        """Render the current state as an image."""
        return self.renderer.render(state)

    def reset(self, key=None) -> Tuple[AirRaidObservation, AirRaidState]:
        """
        Resets the game state to the initial state.

        Returns:
            The initial observation and state
        """
        # Initialize building positions
        building_x = jnp.array([
                -BUILDING_WIDTH,
                -BUILDING_WIDTH + BUILDING_SPACING
        ])
        building_y = jnp.array([BUILDING_INITIAL_Y, BUILDING_INITIAL_Y])
        building_damage = jnp.zeros(NUM_BUILDINGS, dtype=jnp.int32)

        # Initialize enemy arrays (all inactive initially)
        enemy_x = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_y = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_type = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_active = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_has_fired = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)  # Track firing status

        # Initialize missile arrays (all inactive initially)
        player_missile_x = jnp.zeros(NUM_PLAYER_MISSILES, dtype=jnp.int32)
        player_missile_y = jnp.zeros(NUM_PLAYER_MISSILES, dtype=jnp.int32)
        player_missile_active = jnp.zeros(NUM_PLAYER_MISSILES, dtype=jnp.int32)

        enemy_missile_x = jnp.zeros(NUM_ENEMY_MISSILES, dtype=jnp.int32)
        enemy_missile_y = jnp.zeros(NUM_ENEMY_MISSILES, dtype=jnp.int32)
        enemy_missile_active = jnp.zeros(NUM_ENEMY_MISSILES, dtype=jnp.int32)

        # Initialize random key
        rng = random.PRNGKey(0)
        if key is not None: # Allow passing a key for reproducibility
            rng = key

        state = AirRaidState(
            player_x=jnp.array(PLAYER_INITIAL_X),
            player_y=jnp.array(PLAYER_INITIAL_Y),
            player_lives=jnp.array(3),
            building_x=building_x,
            building_y=building_y,
            building_damage=building_damage,
            enemy_x=enemy_x,
            enemy_y=enemy_y,
            enemy_type=enemy_type,
            enemy_active=enemy_active,
            enemy_has_fired=enemy_has_fired,
            player_missile_x=player_missile_x,
            player_missile_y=player_missile_y,
            player_missile_active=player_missile_active,
            enemy_missile_x=enemy_missile_x,
            enemy_missile_y=enemy_missile_y,
            enemy_missile_active=enemy_missile_active,
            score=jnp.array(0),
            step_counter=jnp.array(0),
            flash_counter=jnp.array(0),  # Initialize flash counter
            rng=rng,
        )

        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AirRaidState, action: chex.Array) -> Tuple[AirRaidObservation, AirRaidState, float, bool, AirRaidInfo]:
        """
        Steps the game state forward by one frame.
        Args: state: Current game state, action: Action to take
        Returns: Updated game state, observation, reward, done flag, and info
        """

        # Update building positions
        new_building_x = state.building_x + BUILDING_VELOCITY
        new_building_x = jnp.where(
            new_building_x > WIDTH,
            new_building_x - (WIDTH + BUILDING_WIDTH + 10),
            new_building_x
        )

        # Update player position
        new_player_x = player_step(state.player_x, action)

        # Spawn new enemies
        new_enemy_x, new_enemy_y, new_enemy_type, new_enemy_active, new_enemy_has_fired, new_rng = spawn_enemy(state._replace(player_x=new_player_x))

        # Update existing enemies
        updated_enemy_y, updated_enemy_active, updated_enemy_has_fired, updated_building_damage = update_enemies(
            state._replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=new_enemy_active,
                enemy_has_fired=new_enemy_has_fired,
                rng=new_rng
            )
        )

        # Handle player firing missiles
        new_player_missile_x, new_player_missile_y, new_player_missile_active = fire_player_missile(
            state._replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=updated_enemy_active,
                building_damage=updated_building_damage
            ),
            action
        )

        # Handle enemy firing missiles
        new_enemy_missile_x, new_enemy_missile_y, new_enemy_missile_active, updated_enemy_has_fired, newer_rng = fire_enemy_missiles(
            state._replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=updated_enemy_active,
                enemy_has_fired=updated_enemy_has_fired,
                building_damage=updated_building_damage,
                player_missile_x=new_player_missile_x,
                player_missile_y=new_player_missile_y,
                player_missile_active=new_player_missile_active,
                rng=new_rng
            )
        )

        # Update missile positions
        updated_player_missile_y, updated_player_missile_active, updated_enemy_missile_y, updated_enemy_missile_active = update_missiles(
            state._replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=updated_enemy_active,
                building_damage=updated_building_damage,
                player_missile_x=new_player_missile_x,
                player_missile_y=new_player_missile_y,
                player_missile_active=new_player_missile_active,
                enemy_missile_x=new_enemy_missile_x,
                enemy_missile_y=new_enemy_missile_y,
                enemy_missile_active=new_enemy_missile_active,
                rng=newer_rng
            )
        )

        # Detect and handle collisions
        final_enemy_active, final_player_missile_active, final_enemy_missile_active, new_score, new_player_lives, final_building_damage = detect_collisions(
            state._replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=updated_enemy_active,
                building_damage=updated_building_damage,
                player_missile_y=updated_player_missile_y,
                player_missile_active=updated_player_missile_active,
                enemy_missile_x=new_enemy_missile_x,
                enemy_missile_y=updated_enemy_missile_y,
                enemy_missile_active=updated_enemy_missile_active  # Fixed: use updated_enemy_missile_active instead of new_enemy_missile_active
            )
        )

        # Create the new state first
        new_state = state._replace(
            player_x=new_player_x,
            player_y=state.player_y,
            player_lives=new_player_lives,
            building_x=new_building_x,
            building_y=state.building_y,
            building_damage=final_building_damage,
            enemy_x=new_enemy_x,
            enemy_y=updated_enemy_y,
            enemy_type=new_enemy_type,
            enemy_active=final_enemy_active,
            enemy_has_fired=updated_enemy_has_fired,
            player_missile_x=new_player_missile_x,
            player_missile_y=updated_player_missile_y,
            player_missile_active=final_player_missile_active,
            enemy_missile_x=new_enemy_missile_x,
            enemy_missile_y=updated_enemy_missile_y,
            enemy_missile_active=final_enemy_missile_active,
            score=new_score,
            step_counter=state.step_counter + 1,
            rng=newer_rng,
        )

        # Check if game should be over (but not counting flash animation)
        should_be_game_over = self._should_be_game_over(new_state)

        # Check if any building was completely destroyed (reached MAX_BUILDING_DAMAGE)
        building_was_destroyed = jnp.any(
            jnp.logical_and(
                final_building_damage >= MAX_BUILDING_DAMAGE,  # New damage is at max
                state.building_damage < MAX_BUILDING_DAMAGE    # Old damage was less than max
            )
        )

        # Start flash sequence if building was destroyed OR game is over
        should_start_flash = jnp.logical_or(building_was_destroyed, should_be_game_over)

        # Flash for 20 frames (4 flashes: each flash is 5 frames, alternating on/off)
        flash_counter = jnp.where(
            should_start_flash,
            jnp.where(new_state.flash_counter == 0, 1, new_state.flash_counter + 1),  # Start or continue flashing
            jnp.where(new_state.flash_counter > 0, new_state.flash_counter + 1, 0)   # Continue countdown if already flashing
        )

        # Reset flash counter when done (after 20 frames)
        flash_counter = jnp.where(flash_counter > 20, 0, flash_counter)

        # Update the state with the new flash counter
        new_state = new_state._replace(flash_counter=flash_counter)


        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        def do_reset(_):
            obs, reset_state = self.reset(new_state.rng)
            return obs, reset_state, env_reward, False, info

        def no_reset(_):
            return observation, new_state, env_reward, done, info

        return jax.lax.cond(done, do_reset, no_reset, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AirRaidState) -> AirRaidObservation:
        """
        Transforms the raw state into an observation.
        Args: Current game state
        Returns: Observation object containing entity positions and game data
        """
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(PLAYER_HEIGHT, dtype=jnp.int32)
        )

        def _get_building_obs(i):
            height = BUILDING_HEIGHTS[state.building_damage[i]]
            y_pos = BUILDING_Y_POSITIONS[state.building_damage[i]]
            return jnp.array([state.building_x[i], y_pos, BUILDING_WIDTH, height], dtype=jnp.int32)
        buildings_arr = jax.vmap(_get_building_obs)(jnp.arange(NUM_BUILDINGS))

        def _get_enemy_obs(i):
            width = jnp.select(
                [state.enemy_type[i] == 0, state.enemy_type[i] < 3, state.enemy_type[i] == 3],
                [16, 14, 14], default=0
            )
            height = jnp.select(
                [state.enemy_type[i] == 0, state.enemy_type[i] < 3, state.enemy_type[i] == 3],
                [18, 16, 14], default=0
            )
            active = state.enemy_active[i] == 1
            x = jnp.where(active, state.enemy_x[i], -1)
            y = jnp.where(active, state.enemy_y[i], -1)
            w = jnp.where(active, width, 0)
            h = jnp.where(active, height, 0)
            enemy_type = jnp.where(active, state.enemy_type[i], -1)
            return jnp.array([x, y, w, h, enemy_type], dtype=jnp.int32)
        enemies_arr = jax.vmap(_get_enemy_obs)(jnp.arange(TOTAL_ENEMIES))

        def _get_player_missile_obs(i):
            active = state.player_missile_active[i] == 1
            x = jnp.where(active, state.player_missile_x[i], -1)
            y = jnp.where(active, state.player_missile_y[i], -1)
            w = jnp.where(active, MISSILE_WIDTH, 0)
            h = jnp.where(active, MISSILE_HEIGHT, 0)
            return jnp.array([x, y, w, h], dtype=jnp.int32)
        player_missiles_arr = jax.vmap(_get_player_missile_obs)(jnp.arange(NUM_PLAYER_MISSILES))

        def _get_enemy_missile_obs(i):
            active = state.enemy_missile_active[i] == 1
            x = jnp.where(active, state.enemy_missile_x[i], -1)
            y = jnp.where(active, state.enemy_missile_y[i], -1)
            w = jnp.where(active, MISSILE_WIDTH, 0)
            h = jnp.where(active, MISSILE_HEIGHT, 0)
            return jnp.array([x, y, w, h], dtype=jnp.int32)
        enemy_missiles_arr = jax.vmap(_get_enemy_missile_obs)(jnp.arange(NUM_ENEMY_MISSILES))

        return AirRaidObservation(
            player=player,
            buildings=buildings_arr,
            enemies=enemies_arr,
            player_missiles=player_missiles_arr,
            enemy_missiles=enemy_missiles_arr,
            score=state.score,
            lives=state.player_lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AirRaidObservation) -> jnp.ndarray:
        """
        Converts the observation to a flat array for RL algorithms.
        Args: obs: Observation object
        Returns:  Flattened array representation of the observation
        """
        leaves, _ = jax.tree_util.tree_flatten(obs)
        return jnp.concatenate([leaf.flatten() for leaf in leaves])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        player_space = spaces.Dict({
            "x": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
            "y": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
            "width": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
            "height": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
        })
        # For entities: [x, y, w, h, (type)]
        # Inactive entities are (-1, -1, 0, 0, -1), so low bound is -1
        buildings_space = spaces.Box(
            low=jnp.array([-WIDTH, -1, 0, 0]),
            high=jnp.array([WIDTH, HEIGHT, BUILDING_WIDTH, BUILDING_HEIGHT]),
            shape=(NUM_BUILDINGS, 4), dtype=jnp.int32)

        enemies_space = spaces.Box(
            low=jnp.array([-1, -1, 0, 0, -1]),
            high=jnp.array([WIDTH, HEIGHT, 16, 18, 3]), # Max w=16, h=18, type=3
            shape=(TOTAL_ENEMIES, 5), dtype=jnp.int32)

        player_missiles_space = spaces.Box(
            low=jnp.array([-1, -1, 0, 0]),
            high=jnp.array([WIDTH, HEIGHT, MISSILE_WIDTH, MISSILE_HEIGHT]),
            shape=(NUM_PLAYER_MISSILES, 4), dtype=jnp.int32)

        enemy_missiles_space = spaces.Box(
            low=jnp.array([-1, -1, 0, 0]),
            high=jnp.array([WIDTH, HEIGHT, MISSILE_WIDTH, MISSILE_HEIGHT]),
            shape=(NUM_ENEMY_MISSILES, 4), dtype=jnp.int32)


        return spaces.Dict({
            "player": player_space,
            "buildings": buildings_space,
            "enemies": enemies_space,
            "player_missiles": player_missiles_space,
            "enemy_missiles": enemy_missiles_space,
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(HEIGHT, WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AirRaidState, all_rewards: chex.Array = None) -> AirRaidInfo:

        return AirRaidInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: AirRaidState, state: AirRaidState) -> float:
        score_reward = state.score - previous_state.score
        life_penalty = (previous_state.player_lives - state.player_lives) * 25
        return score_reward - life_penalty

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: AirRaidState, state: AirRaidState) -> chex.Array:

        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _should_be_game_over(self, state: AirRaidState) -> bool:
        """Check if game over conditions are met (ignoring flash animation)"""
        # Game is over if player has no lives left
        player_dead = jnp.less_equal(state.player_lives, 0)

        # Game is over if both buildings are completely destroyed (damage >= MAX_BUILDING_DAMAGE)
        buildings_destroyed = jnp.all(state.building_damage >= MAX_BUILDING_DAMAGE)

        return jnp.logical_or(player_dead, buildings_destroyed)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AirRaidState) -> bool:
        # Game over conditions are met
        game_over_conditions = self._should_be_game_over(state)

        # If game over conditions are met, wait for flash animation to complete
        # Flash 4 times (20 frames total)
        flash_complete = state.flash_counter == 0  # Flash counter resets to 0 when done

        # Game is done when game over conditions are met AND flash animation is complete
        return jnp.logical_and(game_over_conditions, flash_complete)


class AirRaidRenderer(JAXGameRenderer):
    def __init__(self, consts: AirRaidConstants = None):
        super().__init__()
        self.consts = consts or AirRaidConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_BUILDING,
            self.SPRITE_ENEMY25,
            self.SPRITE_ENEMY50,
            self.SPRITE_ENEMY75,
            self.SPRITE_ENEMY100,
            self.SPRITE_MISSILE,
            self.SPRITE_LIFE,
            self.DIGIT_SPRITES
        ) = self.load_sprites()

    def load_sprites(self):
        """Load all sprites required for AirRaid rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load sprites
        player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/player.npy"))
        building = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/building.npy"))
        enemy25 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy25.npy"))
        enemy50 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy50.npy"))
        enemy75 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy75.npy"))
        enemy100 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy100.npy"))
        missile = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/missile.npy"))
        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/background.npy"))
        life = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/life.npy"))

        # Convert all sprites to the expected format (add frame dimension)
        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
        SPRITE_BUILDING = jnp.expand_dims(building, axis=0)
        SPRITE_ENEMY25 = jnp.expand_dims(enemy25, axis=0)
        SPRITE_ENEMY50 = jnp.expand_dims(enemy50, axis=0)
        SPRITE_ENEMY75 = jnp.expand_dims(enemy75, axis=0)
        SPRITE_ENEMY100 = jnp.expand_dims(enemy100, axis=0)
        SPRITE_MISSILE = jnp.expand_dims(missile, axis=0)
        SPRITE_LIFE = jnp.expand_dims(life, axis=0)

        # Load digits for scores
        DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/airraid/score_{}.npy"),
            num_chars=11
        )

        return (
            SPRITE_BG,
            SPRITE_PLAYER,
            SPRITE_BUILDING,
            SPRITE_ENEMY25,
            SPRITE_ENEMY50,
            SPRITE_ENEMY75,
            SPRITE_ENEMY100,
            SPRITE_MISSILE,
            SPRITE_LIFE,
            DIGIT_SPRITES
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state_or_obs):
        is_observation = isinstance(state_or_obs, AirRaidObservation)

        raster = jr.create_initial_frame(width=WIDTH, height=HEIGHT)

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)


        if is_observation:
            def render_building(i, raster_in):
                x, y, w, h = state_or_obs.buildings[i]
                is_active = h > 0
                frame = jr.get_sprite_frame(self.SPRITE_BUILDING, 0)
                render_result = jr.render_at(raster_in, x, y, frame)
                return jnp.where(is_active, render_result, raster_in)

            raster = jax.lax.fori_loop(0, NUM_BUILDINGS, render_building, raster)
        else: # is state
            def render_building(i, raster_in):
                frame_building = jr.get_sprite_frame(self.SPRITE_BUILDING, 0)
                damage_level = state_or_obs.building_damage[i]
                building_x = state_or_obs.building_x[i]
                building_y = BUILDING_Y_POSITIONS[damage_level]
                return jr.render_at(raster_in, building_x, building_y, frame_building)

            raster = jax.lax.fori_loop(0, NUM_BUILDINGS, render_building, raster)

        if is_observation:
            def render_enemy(i, raster_in):
                x, y, w, h, type = state_or_obs.enemies[i]
                is_active = w > 0

                sprite_25 = jr.get_sprite_frame(self.SPRITE_ENEMY25, 0)
                sprite_50 = jr.get_sprite_frame(self.SPRITE_ENEMY50, 0)
                sprite_75 = jr.get_sprite_frame(self.SPRITE_ENEMY75, 0)
                sprite_100 = jr.get_sprite_frame(self.SPRITE_ENEMY100, 0)

                enemy_sprite = jnp.select(
                    [type == 0, type == 1, type == 2, type == 3],
                    [sprite_25, sprite_50, sprite_75, sprite_100],
                    default=sprite_25 # Default should not be used for active enemies
                )

                render_result = jr.render_at(raster_in, x, y, enemy_sprite)
                return jnp.where(is_active, render_result, raster_in)

            raster = jax.lax.fori_loop(0, TOTAL_ENEMIES, render_enemy, raster)
        else: # is state
            def render_enemy(i, raster_in):
                is_active = state_or_obs.enemy_active[i] == 1
                is_type0 = state_or_obs.enemy_type[i] == 0
                is_type1 = state_or_obs.enemy_type[i] == 1
                is_type2 = state_or_obs.enemy_type[i] == 2

                sprite_25 = jr.get_sprite_frame(self.SPRITE_ENEMY25, 0)
                sprite_50 = jr.get_sprite_frame(self.SPRITE_ENEMY50, 0)
                sprite_75 = jr.get_sprite_frame(self.SPRITE_ENEMY75, 0)
                sprite_100 = jr.get_sprite_frame(self.SPRITE_ENEMY100, 0)

                enemy_sprite = jnp.where(is_type0, sprite_25,
                            jnp.where(is_type1, sprite_50,
                            jnp.where(is_type2, sprite_75, sprite_100)))

                render_result = jr.render_at(raster_in, state_or_obs.enemy_x[i], state_or_obs.enemy_y[i], enemy_sprite)
                return jnp.where(is_active, render_result, raster_in)

            raster = jax.lax.fori_loop(0, TOTAL_ENEMIES, render_enemy, raster)

        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        if is_observation:
            raster = jr.render_at(raster, state_or_obs.player.x, state_or_obs.player.y, frame_player)
        else:
            raster = jr.render_at(raster, state_or_obs.player_x, state_or_obs.player_y, frame_player)

        if is_observation:
            def render_player_missile(i, raster_in):
                x, y, w, h = state_or_obs.player_missiles[i]
                is_active = w > 0
                frame_missile = jr.get_sprite_frame(self.SPRITE_MISSILE, 0)
                render_result = jr.render_at(raster_in, x, y, frame_missile)
                return jnp.where(is_active, render_result, raster_in)

            raster = jax.lax.fori_loop(0, NUM_PLAYER_MISSILES, render_player_missile, raster)
        else: # is state
            def render_player_missile(i, raster_in):
                frame_missile = jr.get_sprite_frame(self.SPRITE_MISSILE, 0)
                render_result = jr.render_at(raster_in, state_or_obs.player_missile_x[i],
                                            state_or_obs.player_missile_y[i], frame_missile)
                return jnp.where(state_or_obs.player_missile_active[i] == 1, render_result, raster_in)

            raster = jax.lax.fori_loop(0, NUM_PLAYER_MISSILES, render_player_missile, raster)

        if is_observation:
            def render_enemy_missile(i, raster_in):
                x, y, w, h = state_or_obs.enemy_missiles[i]
                is_active = w > 0
                frame_missile = jr.get_sprite_frame(self.SPRITE_MISSILE, 0)
                render_result = jr.render_at(raster_in, x, y, frame_missile)
                return jnp.where(is_active, render_result, raster_in)

            raster = jax.lax.fori_loop(0, NUM_ENEMY_MISSILES, render_enemy_missile, raster)
        else: # is state
            def render_enemy_missile(i, raster_in):
                frame_missile = jr.get_sprite_frame(self.SPRITE_MISSILE, 0)
                render_result = jr.render_at(raster_in, state_or_obs.enemy_missile_x[i],
                                            state_or_obs.enemy_missile_y[i], frame_missile)
                return jnp.where(state_or_obs.enemy_missile_active[i] == 1, render_result, raster_in)

            raster = jax.lax.fori_loop(0, NUM_ENEMY_MISSILES, render_enemy_missile, raster)

        # Add a black bar at the bottom of the screen
        black_bar_height = 20
        black_bar_y = HEIGHT - black_bar_height
        # raster = raster.at[:, black_bar_y:, :].set(0)
        raster = raster.at[black_bar_y:, :, :].set(0)  # Swapped indices to match pre-transpose orientation

        score_value = state_or_obs.score
        score_y = 5
        score_x_start = 30
        max_digits_render = 6
        padded_digits_render = jnp.zeros(max_digits_render, dtype=jnp.int32)

        def get_digits_body(i, val):
            digits_array, current_score = val
            digit_index = max_digits_render - 1 - i
            digit = current_score % 10
            digits_array = digits_array.at[digit_index].set(digit)
            current_score = current_score // 10
            return digits_array, current_score

        padded_digits_render, _ = jax.lax.fori_loop(
            0, max_digits_render, get_digits_body, (padded_digits_render, score_value)
        )

        # Determine which digits should be visible based on score value
        is_score_zero = (score_value == 0)
        indices = jnp.arange(max_digits_render)

        is_significant = (padded_digits_render > 0)
        temp_indices = jnp.where(is_significant, indices, max_digits_render)
        first_significant_idx = jnp.min(temp_indices)

        visible_if_zero = (indices == max_digits_render - 1)
        visible_if_nonzero = (indices >= first_significant_idx)

        should_be_visible = jnp.where(is_score_zero, visible_if_zero, visible_if_nonzero)

        invisible_digit_index = 10
        final_digits_to_render = jnp.where(
            should_be_visible,
            padded_digits_render,
            invisible_digit_index
        )

        raster = jr.render_label(
            raster,
            score_x_start,
            score_y,
            final_digits_to_render,
            self.DIGIT_SPRITES
        )

        lives = state_or_obs.player_lives if not is_observation else state_or_obs.lives
        def render_life(i, raster_in):
            life_sprite = jr.get_sprite_frame(self.SPRITE_LIFE, 0)
            life_width = life_sprite.shape[0]
            life_spacing = life_width + 3

            life_start_x = 30
            life_y = 200

            icon_x = life_start_x + i * life_spacing

            result = jr.render_at(raster_in, icon_x, life_y, life_sprite)

            # Show life indicator if we have more than i+1 lives (so 3 lives shows 2 indicators)
            return jnp.where(i < lives - 1, result, raster_in)

        raster = jax.lax.fori_loop(0, 2, render_life, raster)  # Only render 2 life indicators (0, 1)

        # Apply simple white flash effect for building damage and game over
        if not is_observation:  # Only apply flash to state rendering
            # Check if we should be flashing (flash counter > 0)
            should_flash = state_or_obs.flash_counter > 0

            # Flash every 5 frames (white for 5, normal for 5) to get 4 quick flashes in 20 frames
            flash_on = (state_or_obs.flash_counter % 10) < 5

            # Create white screen
            white_raster = jnp.full_like(raster, 255)

            # Show white screen when flashing
            raster = jnp.where(jnp.logical_and(should_flash, flash_on), white_raster, raster)

        # return raster.transpose(1, 0, 2)  # Convert to (H, W, C) format
        return raster

if __name__ == "__main__":
    pygame.init()
    WINDOW_WIDTH = 160 * 3
    WINDOW_HEIGHT = 210 * 3
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Air Raid Game")
    clock = pygame.time.Clock()


    game = JaxAirRaid(frameskip=1)
    renderer = game.renderer

    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                if frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

                        if done:
                           obs, curr_state = jitted_reset(curr_state.rng)


        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)
                if done:
                    obs, curr_state = jitted_reset(curr_state.rng)
        raster = renderer.render(curr_state)
        jr.update_pygame(screen, raster, 3, WIDTH, HEIGHT)


        counter += 1
        clock.tick(60)

    pygame.quit()
