import os
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as random
import chex
from flax import struct
import jaxatari.spaces as spaces

from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer

class AirRaidConstants(struct.PyTreeNode):
    # Game environment
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=250)

    # Player
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=14)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=12)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=3)
    PLAYER_INITIAL_X: int = struct.field(pytree_node=False, default=80)
    PLAYER_INITIAL_Y: int = struct.field(pytree_node=False, default=140)
    MAX_PLAYER_LIVES: int = struct.field(pytree_node=False, default=4)

    # Buildings
    NUM_BUILDINGS: int = struct.field(pytree_node=False, default=2)
    BUILDING_WIDTH: int = struct.field(pytree_node=False, default=50)
    BUILDING_HEIGHT: int = struct.field(pytree_node=False, default=25)
    MAX_BUILDING_DAMAGE: int = struct.field(pytree_node=False, default=6)
    BUILDING_INITIAL_Y: int = struct.field(pytree_node=False, default=205)
    BUILDING_VELOCITY: int = struct.field(pytree_node=False, default=1)
    BUILDING_SPACING: int = struct.field(pytree_node=False, default=90)

    # Height and Y position based on damage level (bottom of building anchored at y=230)
    BUILDING_HEIGHTS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([25, 21, 17, 13, 9, 5, 0]),
    )
    BUILDING_Y_POSITIONS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([205, 209, 213, 217, 221, 225, 230]),
    )

    # Enemies
    NUM_ENEMIES_PER_TYPE: int = struct.field(pytree_node=False, default=3)
    TOTAL_ENEMIES: int = struct.field(pytree_node=False, default=9)  # NUM_ENEMIES_PER_TYPE * 4
    MAX_ACTIVE_ENEMIES: int = struct.field(pytree_node=False, default=3)
    ENEMY_INITIAL_Y: int = struct.field(pytree_node=False, default=69)
    ENEMY_SLOW_SPEED: int = struct.field(pytree_node=False, default=1)
    ENEMY_FAST_SPEED: int = struct.field(pytree_node=False, default=2)
    ENEMY_SPAWN_Y: int = struct.field(pytree_node=False, default=30)
    ENEMY_SPAWN_PROB: float = struct.field(pytree_node=False, default=0.04)

    # Missiles
    MISSILE_WIDTH: int = struct.field(pytree_node=False, default=2)
    MISSILE_HEIGHT: int = struct.field(pytree_node=False, default=2)
    NUM_PLAYER_MISSILES: int = struct.field(pytree_node=False, default=1)
    NUM_ENEMY_MISSILES: int = struct.field(pytree_node=False, default=1)
    PLAYER_MISSILE_SPEED: int = struct.field(pytree_node=False, default=-6)
    ENEMY_MISSILE_SPEED: int = struct.field(pytree_node=False, default=4)
    ENEMY_FIRE_PROB: float = struct.field(pytree_node=False, default=0.05)


DEFAULT_AIRRAID_CONSTANTS = AirRaidConstants()

# Immutable state container
class AirRaidState(struct.PyTreeNode):
    player_x: chex.Array = struct.field()
    player_y: chex.Array = struct.field()
    player_lives: chex.Array = struct.field()
    player_visible: chex.Array = struct.field()

    building_x: chex.Array = struct.field()
    building_y: chex.Array = struct.field()
    building_damage: chex.Array = struct.field()

    enemy_x: chex.Array = struct.field()
    enemy_y: chex.Array = struct.field()
    enemy_type: chex.Array = struct.field()
    enemy_active: chex.Array = struct.field()
    enemy_has_fired: chex.Array = struct.field()  # Track which enemies have already fired

    player_missile_x: chex.Array = struct.field()
    player_missile_y: chex.Array = struct.field()
    player_missile_active: chex.Array = struct.field()

    enemy_missile_x: chex.Array = struct.field()
    enemy_missile_y: chex.Array = struct.field()
    enemy_missile_active: chex.Array = struct.field()

    score: chex.Array = struct.field()
    step_counter: chex.Array = struct.field()
    flash_counter: chex.Array = struct.field()  # Counter for screen flashing animation (building damage or game over)
    rng: chex.Array = struct.field()  # Random key for stochastic game elements

class AirRaidObservation(struct.PyTreeNode):
    player: ObjectObservation = struct.field()
    buildings: ObjectObservation = struct.field()
    enemies: ObjectObservation = struct.field()
    player_missiles: ObjectObservation = struct.field()
    enemy_missiles: ObjectObservation = struct.field()
    score: jnp.ndarray = struct.field()
    lives: jnp.ndarray = struct.field()

class AirRaidInfo(struct.PyTreeNode):
    time: jnp.ndarray = struct.field()

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
    RIGHT_BOUNDARY = AirRaidConstants.WIDTH - AirRaidConstants.PLAYER_WIDTH - 10

    # Check if left or right button was pressed
    move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
    move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

    player_x = jnp.where(
        move_left,
        jnp.maximum(player_x - AirRaidConstants.PLAYER_SPEED, LEFT_BOUNDARY),
        player_x
    )

    player_x = jnp.where(
        move_right,
        jnp.minimum(player_x + AirRaidConstants.PLAYER_SPEED, RIGHT_BOUNDARY),
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
    first_inactive = jnp.max(jnp.where(inactive_mask, jnp.arange(AirRaidConstants.TOTAL_ENEMIES), -1))
    active_enemy_count = jnp.sum(state.enemy_active)

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
        random.randint(pos_key1, shape=(), minval=10, maxval=AirRaidConstants.WIDTH - 30),
        random.randint(pos_key2, shape=(), minval=10, maxval=AirRaidConstants.WIDTH - 30)
    ])

    # Check overlaps and select first valid position
    overlaps = jnp.array([has_overlap(candidates[0]), has_overlap(candidates[1])])
    valid_candidates = ~overlaps
    new_x = jnp.where(valid_candidates[0], candidates[0], candidates[1])

    # Spawn conditions: probability + slot available + under active cap + at least one valid position
    should_spawn = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(spawn_prob < AirRaidConstants.ENEMY_SPAWN_PROB, first_inactive >= 0),
            active_enemy_count < AirRaidConstants.MAX_ACTIVE_ENEMIES,
        ),
        jnp.any(valid_candidates)
    )

    # Update enemy arrays
    enemy_x = state.enemy_x.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(new_x), state.enemy_x[first_inactive]))
    enemy_y = state.enemy_y.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(AirRaidConstants.ENEMY_SPAWN_Y), state.enemy_y[first_inactive]))
    enemy_type = state.enemy_type.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(new_type), state.enemy_type[first_inactive]))
    enemy_active = state.enemy_active.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(1), state.enemy_active[first_inactive]))
    enemy_has_fired = state.enemy_has_fired.at[first_inactive].set(jnp.where(should_spawn, jnp.int32(0), state.enemy_has_fired[first_inactive]))  # Reset firing status

    return enemy_x, enemy_y, enemy_type, enemy_active, enemy_has_fired, rng

@jax.jit
def update_enemies(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Updates all enemy positions. Enemies move down the screen."""
    # Extract state components
    enemy_y = state.enemy_y
    enemy_type = state.enemy_type
    enemy_active = state.enemy_active
    enemy_has_fired = state.enemy_has_fired
    building_damage = state.building_damage

    # Move active enemies down

    # 0, 1 slow: rakete und heli, flieger und ufo waren schnell
    # 0, 3 slow: ufo und heli
    # -> 0 ist heli, 1 ist rakete, 2 ist flieger, 3 ist ufo
    enemy_speeds = jnp.where(
        jnp.logical_or(enemy_type == 1, enemy_type == 3), # slow are: rocket and ufo
        AirRaidConstants.ENEMY_SLOW_SPEED,
        AirRaidConstants.ENEMY_FAST_SPEED,
    )
    enemy_y = jnp.where(enemy_active == 1, enemy_y + enemy_speeds, enemy_y)

    # Deactivate enemies that reach the bottom
    reached_player = enemy_y > jnp.int32(AirRaidConstants.PLAYER_INITIAL_Y - 20)  # Changed from HEIGHT to PLAYER_INITIAL_Y
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
    inactive_indices = jnp.where(inactive_missile_mask, jnp.arange(AirRaidConstants.NUM_PLAYER_MISSILES), -1)
    first_inactive = jnp.max(inactive_indices)

    # Only fire if button pressed and missile slot is available
    should_fire = jnp.logical_and(is_fire, first_inactive >= 0)

    missile_x = state.player_x + (AirRaidConstants.PLAYER_WIDTH // 2) - (AirRaidConstants.MISSILE_WIDTH // 2)


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
    inactive_indices = jnp.where(inactive_missile_mask, jnp.arange(AirRaidConstants.NUM_ENEMY_MISSILES), -1)
    first_inactive = jnp.max(inactive_indices)  # Get the highest valid index

    # Generate random values for firing decision and which enemy fires
    rng, fire_key, enemy_key = random.split(rng, 3)
    fire_prob = random.uniform(fire_key)

    # Count active enemies without using nonzero
    active_enemy_count = jnp.sum(state.enemy_active)

    # Randomly select an enemy index (0 to TOTAL_ENEMIES-1)
    random_enemy_idx = random.randint(enemy_key, shape=(), minval=0, maxval=AirRaidConstants.TOTAL_ENEMIES)

    # We'll iterate through the enemies and select the first active one after our random index
    # This is a workaround since we can't use jnp.nonzero in jitted code

    # This function finds a valid active enemy that hasn't fired yet
    def find_active_unfired_enemy(random_idx, enemy_active, enemy_has_fired):
        # Create a shifted array where we start checking from random_idx
        indices = (random_idx + jnp.arange(AirRaidConstants.TOTAL_ENEMIES)) % AirRaidConstants.TOTAL_ENEMIES

        # For each index, check if it's active AND hasn't fired yet
        can_fire = jnp.logical_and(enemy_active[indices] == 1, enemy_has_fired[indices] == 0)

        # Compute scores - unfired active enemies get high scores
        scores = jnp.where(
            can_fire,
            AirRaidConstants.TOTAL_ENEMIES - jnp.arange(AirRaidConstants.TOTAL_ENEMIES),
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
    selected_enemy_active = state.enemy_active[firing_enemy_idx] == 1
    enemy_hasnt_fired = state.enemy_has_fired[firing_enemy_idx] == 0
    no_active_missiles = jnp.sum(enemy_missile_active) == 0  # Ensure no missiles are currently active
    player_targetable = jnp.logical_and(state.player_visible == 1, state.player_lives > 0)

    can_fire = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(fire_prob < AirRaidConstants.ENEMY_FIRE_PROB, first_inactive >= 0),
            jnp.logical_and(enemy_available, jnp.logical_and(selected_enemy_active, enemy_hasnt_fired))
        ),
        jnp.logical_and(no_active_missiles, player_targetable)
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
        state.player_missile_y + AirRaidConstants.PLAYER_MISSILE_SPEED,
        state.player_missile_y
    )

    # Move enemy missiles down
    enemy_missile_y = jnp.where(
        state.enemy_missile_active == 1,
        state.enemy_missile_y + AirRaidConstants.ENEMY_MISSILE_SPEED,
        state.enemy_missile_y
    )

    # Deactivate missiles that go off-screen
    player_missile_active = jnp.where(
        player_missile_y < 0,
        0,
        state.player_missile_active
    )

    enemy_missile_active = jnp.where(
        enemy_missile_y > AirRaidConstants.HEIGHT,
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

    score_values = jnp.array([25, 50, 75, 100], dtype=jnp.int32)
    enemy_widths = jnp.where(state.enemy_type == 0, 16, 14)
    enemy_heights = jnp.where(state.enemy_type == 0, 18, jnp.where(state.enemy_type < 3, 16, 14))

    def process_player_missile(carry, pm):
        carry_enemy_active, carry_player_missile_active, carry_score = carry
        is_missile_active = carry_player_missile_active[pm] == 1
        missile_x = state.player_missile_x[pm]
        missile_y = state.player_missile_y[pm]

        def enemy_collision_fn(ex, ey, ew, eh, ea):
            collision = jnp.logical_and(
                jnp.logical_and(
                    missile_x < ex + ew,
                    missile_x + AirRaidConstants.MISSILE_WIDTH > ex,
                ),
                jnp.logical_and(
                    missile_y < ey + eh,
                    missile_y + AirRaidConstants.MISSILE_HEIGHT > ey,
                ),
            )
            return jnp.logical_and(jnp.logical_and(collision, is_missile_active), ea == 1)

        effective_collisions = jax.vmap(enemy_collision_fn)(
            state.enemy_x,
            state.enemy_y,
            enemy_widths,
            enemy_heights,
            carry_enemy_active,
        )

        carry_enemy_active = jnp.where(effective_collisions, 0, carry_enemy_active)
        carry_player_missile_active = carry_player_missile_active.at[pm].set(
            jnp.where(jnp.any(effective_collisions), 0, carry_player_missile_active[pm])
        )
        score_add = jnp.sum(jnp.where(effective_collisions, score_values[state.enemy_type], 0))
        carry_score = carry_score + score_add

        return (carry_enemy_active, carry_player_missile_active, carry_score), None

    (enemy_active, player_missile_active, score), _ = jax.lax.scan(
        process_player_missile,
        (enemy_active, player_missile_active, score),
        jnp.arange(AirRaidConstants.NUM_PLAYER_MISSILES),
    )

    def process_enemy_missile(carry, em):
        carry_enemy_missile_active, carry_player_lives, carry_building_damage = carry
        is_missile_active = carry_enemy_missile_active[em] == 1
        missile_x = state.enemy_missile_x[em]
        missile_y = state.enemy_missile_y[em]

        building_y = DEFAULT_AIRRAID_CONSTANTS.BUILDING_Y_POSITIONS[carry_building_damage]
        building_h = DEFAULT_AIRRAID_CONSTANTS.BUILDING_HEIGHTS[carry_building_damage]

        def building_collision_fn(bx, by, bh):
            collision = jnp.logical_and(
                jnp.logical_and(
                    missile_x >= bx,
                    missile_x < bx + AirRaidConstants.BUILDING_WIDTH,
                ),
                jnp.logical_and(
                    missile_y >= by,
                    missile_y < by + bh,
                ),
            )
            return jnp.logical_and(collision, is_missile_active)

        effective_building_collisions = jax.vmap(building_collision_fn)(state.building_x, building_y, building_h)
        carry_building_damage = jnp.minimum(
            carry_building_damage + effective_building_collisions.astype(carry_building_damage.dtype),
            AirRaidConstants.MAX_BUILDING_DAMAGE,
        )

        player_collision = jnp.logical_and(
            jnp.logical_and(
                missile_x < state.player_x + AirRaidConstants.PLAYER_WIDTH,
                missile_x + AirRaidConstants.MISSILE_WIDTH > state.player_x,
            ),
            jnp.logical_and(
                missile_y < state.player_y + AirRaidConstants.PLAYER_HEIGHT,
                missile_y + AirRaidConstants.MISSILE_HEIGHT > state.player_y,
            ),
        )
        effective_player_collision = jnp.logical_and(player_collision, is_missile_active)
        effective_player_collision = jnp.logical_and(effective_player_collision, state.player_visible == 1)

        missile_deactivated = jnp.logical_or(jnp.any(effective_building_collisions), effective_player_collision)
        carry_enemy_missile_active = carry_enemy_missile_active.at[em].set(
            jnp.where(missile_deactivated, 0, carry_enemy_missile_active[em])
        )
        carry_player_lives = jnp.where(effective_player_collision, carry_player_lives - 1, carry_player_lives)

        return (carry_enemy_missile_active, carry_player_lives, carry_building_damage), None

    (enemy_missile_active, player_lives, building_damage), _ = jax.lax.scan(
        process_enemy_missile,
        (enemy_missile_active, player_lives, building_damage),
        jnp.arange(AirRaidConstants.NUM_ENEMY_MISSILES),
    )

    return enemy_active, player_missile_active, enemy_missile_active, score, player_lives, building_damage


class JaxAirRaid(JaxEnvironment[AirRaidState, AirRaidObservation, AirRaidInfo, AirRaidConstants]):
    # Minimal ALE action set for Pong:
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )
    def __init__(self, consts: AirRaidConstants = None, frameskip: int = 0, reward_funcs: list = None):
        consts = consts or AirRaidConstants()
        super().__init__(consts)
        self.frameskip = frameskip + 1
        if reward_funcs is not None:
            self.reward_funcs = tuple(reward_funcs)
        else:
            self.reward_funcs = None
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
            -AirRaidConstants.BUILDING_WIDTH,
            -AirRaidConstants.BUILDING_WIDTH + AirRaidConstants.BUILDING_SPACING
        ])
        building_y = jnp.array([AirRaidConstants.BUILDING_INITIAL_Y, AirRaidConstants.BUILDING_INITIAL_Y])
        building_damage = jnp.zeros(AirRaidConstants.NUM_BUILDINGS, dtype=jnp.int32)

        # Initialize enemy arrays
        enemy_x = jnp.zeros(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_y = jnp.zeros(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_type = jnp.zeros(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_active = jnp.zeros(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_has_fired = jnp.zeros(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)  # Track firing status

        # Spawn first three enemies together at game start (left-to-right types: 1, 3, 1)
        initial_enemy_x = jnp.array([20, 70, 120], dtype=jnp.int32)
        initial_enemy_types = jnp.array([1, 3, 1], dtype=jnp.int32)
        enemy_x = enemy_x.at[:3].set(initial_enemy_x)
        enemy_y = enemy_y.at[:3].set(jnp.int32(AirRaidConstants.ENEMY_SPAWN_Y))
        enemy_type = enemy_type.at[:3].set(initial_enemy_types)
        enemy_active = enemy_active.at[:3].set(jnp.array([1, 1, 1], dtype=jnp.int32))

        # Initialize missile arrays (all inactive initially)
        player_missile_x = jnp.zeros(AirRaidConstants.NUM_PLAYER_MISSILES, dtype=jnp.int32)
        player_missile_y = jnp.zeros(AirRaidConstants.NUM_PLAYER_MISSILES, dtype=jnp.int32)
        player_missile_active = jnp.zeros(AirRaidConstants.NUM_PLAYER_MISSILES, dtype=jnp.int32)

        enemy_missile_x = jnp.zeros(AirRaidConstants.NUM_ENEMY_MISSILES, dtype=jnp.int32)
        enemy_missile_y = jnp.zeros(AirRaidConstants.NUM_ENEMY_MISSILES, dtype=jnp.int32)
        enemy_missile_active = jnp.zeros(AirRaidConstants.NUM_ENEMY_MISSILES, dtype=jnp.int32)

        # Initialize random key
        rng = random.PRNGKey(0)
        if key is not None: # Allow passing a key for reproducibility
            rng = key

        state = AirRaidState(
            player_x=jnp.array(AirRaidConstants.PLAYER_INITIAL_X),
            player_y=jnp.array(AirRaidConstants.PLAYER_INITIAL_Y),
            player_lives=jnp.array(AirRaidConstants.MAX_PLAYER_LIVES - 1),
            player_visible=jnp.array(1),
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
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))  # Map action index to actual action value

        # Update building positions
        new_building_x = state.building_x + AirRaidConstants.BUILDING_VELOCITY
        new_building_x = jnp.where(
            new_building_x > AirRaidConstants.WIDTH,
            0, 
            new_building_x
        )

        player_is_visible = state.player_visible == 1

        # Update player position
        stepped_player_x = player_step(state.player_x, action)
        new_player_x = jnp.where(player_is_visible, stepped_player_x, state.player_x)

        noop_action = jnp.array(Action.NOOP, dtype=action.dtype)
        effective_action = jnp.where(player_is_visible, action, noop_action)

        # Spawn new enemies only when the player is visible
        pre_spawn_state = state.replace(player_x=new_player_x)
        new_enemy_x, new_enemy_y, new_enemy_type, new_enemy_active, new_enemy_has_fired, new_rng = jax.lax.cond(
            player_is_visible,
            lambda _: spawn_enemy(pre_spawn_state),
            lambda _: (
                pre_spawn_state.enemy_x,
                pre_spawn_state.enemy_y,
                pre_spawn_state.enemy_type,
                pre_spawn_state.enemy_active,
                pre_spawn_state.enemy_has_fired,
                pre_spawn_state.rng,
            ),
            operand=None,
        )

        # Update existing enemies every second frame (effective half-speed with integer per-step speeds)
        enemy_update_state = state.replace(
            player_x=new_player_x,
            enemy_x=new_enemy_x,
            enemy_y=new_enemy_y,
            enemy_type=new_enemy_type,
            enemy_active=new_enemy_active,
            enemy_has_fired=new_enemy_has_fired,
            rng=new_rng,
        )
        should_update_enemies = (state.step_counter % 2) == 0
        updated_enemy_y, updated_enemy_active, updated_enemy_has_fired, updated_building_damage = jax.lax.cond(
            should_update_enemies,
            lambda s: update_enemies(s),
            lambda s: (s.enemy_y, s.enemy_active, s.enemy_has_fired, s.building_damage),
            enemy_update_state,
        )

        # Handle player firing missiles
        new_player_missile_x, new_player_missile_y, new_player_missile_active = fire_player_missile(
            state.replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=updated_enemy_active,
                building_damage=updated_building_damage
            ),
            effective_action
        )

        # Handle enemy firing missiles
        new_enemy_missile_x, new_enemy_missile_y, new_enemy_missile_active, updated_enemy_has_fired, newer_rng = fire_enemy_missiles(
            state.replace(
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
            state.replace(
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
            state.replace(
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

        previous_life_milestone = state.score // 1000
        new_life_milestone = new_score // 1000
        bonus_lives = jnp.maximum(new_life_milestone - previous_life_milestone, 0)
        adjusted_player_lives = jnp.minimum(new_player_lives + bonus_lives, AirRaidConstants.MAX_PLAYER_LIVES)

        life_lost = adjusted_player_lives < state.player_lives
        enemies_remaining = jnp.any(final_enemy_active == 1)
        player_is_alive = adjusted_player_lives > 0
        waiting_for_respawn = jnp.logical_or(state.player_visible == 0, life_lost)

        next_player_visible = jnp.where(
            player_is_alive,
            jnp.where(jnp.logical_and(waiting_for_respawn, enemies_remaining), jnp.int32(0), jnp.int32(1)),
            jnp.int32(0),
        )

        player_reappears = jnp.logical_and(state.player_visible == 0, next_player_visible == 1)
        final_player_x = jnp.where(player_reappears, jnp.int32(AirRaidConstants.PLAYER_INITIAL_X), new_player_x)
        final_player_y = jnp.where(player_reappears, jnp.int32(AirRaidConstants.PLAYER_INITIAL_Y), state.player_y)

        final_player_missile_active = jnp.where(
            life_lost,
            jnp.zeros_like(final_player_missile_active),
            final_player_missile_active,
        )


        # Create the new state first
        new_state = state.replace(
            player_x=final_player_x,
            player_y=final_player_y,
            player_lives=adjusted_player_lives,
            player_visible=next_player_visible,
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
                final_building_damage >= AirRaidConstants.MAX_BUILDING_DAMAGE,  # New damage is at max
                state.building_damage < AirRaidConstants.MAX_BUILDING_DAMAGE    # Old damage was less than max
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
        new_state = new_state.replace(flash_counter=flash_counter)


        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        def do_reset(_):
            obs, reset_state = self.reset(new_state.rng)
            return obs, reset_state, env_reward, done, info

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
        w, h = AirRaidConstants.WIDTH, AirRaidConstants.HEIGHT
        player_is_visible = state.player_visible == 1
        px = jnp.clip(state.player_x, 0, w)
        py = jnp.clip(state.player_y, 0, h)
        player = ObjectObservation.create(
            x=jnp.where(player_is_visible, px, 0),
            y=jnp.where(player_is_visible, py, 0),
            width=jnp.where(player_is_visible, jnp.array(AirRaidConstants.PLAYER_WIDTH, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
            height=jnp.where(player_is_visible, jnp.array(AirRaidConstants.PLAYER_HEIGHT, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
            active=state.player_visible,
        )

        building_active = (state.building_damage < AirRaidConstants.MAX_BUILDING_DAMAGE).astype(jnp.int32)
        building_y = DEFAULT_AIRRAID_CONSTANTS.BUILDING_Y_POSITIONS[state.building_damage]
        building_h = DEFAULT_AIRRAID_CONSTANTS.BUILDING_HEIGHTS[state.building_damage]
        bx = jnp.clip(state.building_x, 0, w)
        by = jnp.clip(building_y, 0, h)
        buildings_obs = ObjectObservation.create(
            x=jnp.where(building_active == 1, bx, 0),
            y=jnp.where(building_active == 1, by, 0),
            width=jnp.full_like(state.building_x, AirRaidConstants.BUILDING_WIDTH),
            height=building_h,
            active=building_active,
            state=state.building_damage,
        )

        enemy_widths = jnp.where(state.enemy_type == 0, 16, 14)
        enemy_heights = jnp.where(
            state.enemy_type == 0,
            18,
            jnp.where(state.enemy_type < 3, 16, 14),
        )
        ex = jnp.clip(state.enemy_x, 0, w)
        ey = jnp.clip(state.enemy_y, 0, h)
        enemies_obs = ObjectObservation.create(
            x=jnp.where(state.enemy_active == 1, ex, 0),
            y=jnp.where(state.enemy_active == 1, ey, 0),
            width=enemy_widths,
            height=enemy_heights,
            active=state.enemy_active,
            visual_id=state.enemy_type,
        )

        pmx = jnp.clip(state.player_missile_x, 0, w)
        pmy = jnp.clip(state.player_missile_y, 0, h)
        player_missiles_obs = ObjectObservation.create(
            x=jnp.where(state.player_missile_active == 1, pmx, 0),
            y=jnp.where(state.player_missile_active == 1, pmy, 0),
            width=jnp.full_like(state.player_missile_x, AirRaidConstants.MISSILE_WIDTH),
            height=jnp.full_like(state.player_missile_y, AirRaidConstants.MISSILE_HEIGHT),
            active=state.player_missile_active,
        )

        emx = jnp.clip(state.enemy_missile_x, 0, w)
        emy = jnp.clip(state.enemy_missile_y, 0, h)
        enemy_missiles_obs = ObjectObservation.create(
            x=jnp.where(state.enemy_missile_active == 1, emx, 0),
            y=jnp.where(state.enemy_missile_active == 1, emy, 0),
            width=jnp.full_like(state.enemy_missile_x, AirRaidConstants.MISSILE_WIDTH),
            height=jnp.full_like(state.enemy_missile_y, AirRaidConstants.MISSILE_HEIGHT),
            active=state.enemy_missile_active,
        )

        return AirRaidObservation(
            player=player,
            buildings=buildings_obs,
            enemies=enemies_obs,
            player_missiles=player_missiles_obs,
            enemy_missiles=enemy_missiles_obs,
            score=state.score,
            lives=state.player_lives
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        player_space = spaces.get_object_space(
            n=None, screen_size=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH)
        )
        buildings_space = spaces.get_object_space(
            n=AirRaidConstants.NUM_BUILDINGS,
            screen_size=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH),
        )
        enemies_space = spaces.get_object_space(
            n=AirRaidConstants.TOTAL_ENEMIES,
            screen_size=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH),
        )
        player_missiles_space = spaces.get_object_space(
            n=AirRaidConstants.NUM_PLAYER_MISSILES,
            screen_size=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH),
        )
        enemy_missiles_space = spaces.get_object_space(
            n=AirRaidConstants.NUM_ENEMY_MISSILES,
            screen_size=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH),
        )


        return spaces.Dict({
            "player": player_space,
            "buildings": buildings_space,
            "enemies": enemies_space,
            "player_missiles": player_missiles_space,
            "enemy_missiles": enemy_missiles_space,
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=AirRaidConstants.MAX_PLAYER_LIVES, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AirRaidState, all_rewards: chex.Array = None) -> AirRaidInfo:

        return AirRaidInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: AirRaidState, state: AirRaidState) -> float:
        score_reward = state.score - previous_state.score
        return score_reward 

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AirRaidState, state: AirRaidState) -> float:
        """Required by the gymnasium wrapper - same as _get_env_reward"""
        return self._get_env_reward(previous_state, state)

    @partial(jax.jit, static_argnums=(0,))
    def _should_be_game_over(self, state: AirRaidState) -> bool:
        """Check if game over conditions are met (ignoring flash animation)"""
        # Game is over if player has no lives left
        player_dead = jnp.less_equal(state.player_lives, 0)

        # Game is over if both buildings are completely destroyed (damage >= MAX_BUILDING_DAMAGE)
        buildings_destroyed = jnp.all(state.building_damage >= AirRaidConstants.MAX_BUILDING_DAMAGE)

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
    def __init__(self, consts: AirRaidConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or AirRaidConstants()

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(AirRaidConstants.HEIGHT, AirRaidConstants.WIDTH),
                channels=3,
                downscale=None,
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "airraid")

        padded_background = self._load_and_pad_background(sprite_path)
        black_bar_sprite = self._create_black_bar_sprite()
        asset_config = [
            {'name': 'background', 'type': 'background', 'data': padded_background},
            {'name': 'player', 'type': 'single', 'file': 'player.npy'},
            {'name': 'building', 'type': 'single', 'file': 'building.npy'},
            {
                'name': 'enemy',
                'type': 'group',
                'files': ['enemy_25_mono.npy', 'enemy_50_mono.npy', 'enemy_75_mono.npy', 'enemy_100_mono.npy']
            },
            {'name': 'missile', 'type': 'single', 'file': 'missile.npy'},
            {'name': 'life', 'type': 'single', 'file': 'life.npy'},
            {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
            {'name': 'black_bar', 'type': 'procedural', 'data': black_bar_sprite},
        ]

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        self.score_digit_spacing = int(self.SHAPE_MASKS["score_digits"].shape[2])
        self.life_spacing = int(self.SHAPE_MASKS["life"].shape[1]) + 3

    def _create_black_bar_sprite(self) -> jnp.ndarray:
        black_bar_height = 20
        black_bar_width = AirRaidConstants.WIDTH
        black_bar = jnp.zeros((black_bar_height, black_bar_width, 4), dtype=jnp.uint8)
        return black_bar.at[:, :, 3].set(255)

    def _load_and_pad_background(self, sprite_path: str) -> jnp.ndarray:
        background = self.jr.loadFrame(os.path.join(sprite_path, "background.npy"))
        current_height, current_width, _ = background.shape

        if current_height < AirRaidConstants.HEIGHT:
            pad_rows = AirRaidConstants.HEIGHT - current_height
            padding = jnp.zeros((pad_rows, current_width, 4), dtype=background.dtype)
            padding = padding.at[:, :, 3].set(255)
            return jnp.concatenate([background, padding], axis=0)

        if current_height > AirRaidConstants.HEIGHT:
            return background[:AirRaidConstants.HEIGHT, :, :]

        return background

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AirRaidState):

        raster = self.jr.create_object_raster(self.BACKGROUND)

        building_mask = self.SHAPE_MASKS["building"]
        enemy_masks = self.SHAPE_MASKS["enemy"]
        player_mask = self.SHAPE_MASKS["player"]
        missile_mask = self.SHAPE_MASKS["missile"]
        life_mask = self.SHAPE_MASKS["life"]
        score_digit_masks = self.SHAPE_MASKS["score_digits"]

        def render_building(i, raster_in):
            damage_level = state.building_damage[i]
            building_x = state.building_x[i]
            building_y = DEFAULT_AIRRAID_CONSTANTS.BUILDING_Y_POSITIONS[damage_level]
            is_active = damage_level < AirRaidConstants.MAX_BUILDING_DAMAGE
            render_result = self.jr.render_at_clipped(raster_in, building_x, building_y, building_mask)
            return jnp.where(is_active, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.NUM_BUILDINGS, render_building, raster)

        def render_enemy(i, raster_in):
            is_active = state.enemy_active[i] == 1
            enemy_type = jnp.clip(state.enemy_type[i], 0, 3)
            enemy_mask = enemy_masks[enemy_type]
            render_result = self.jr.render_at_clipped(raster_in, state.enemy_x[i], state.enemy_y[i], enemy_mask)
            return jnp.where(is_active, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.TOTAL_ENEMIES, render_enemy, raster)

        player_rendered = self.jr.render_at_clipped(raster, state.player_x, state.player_y, player_mask)
        raster = jnp.where(state.player_visible == 1, player_rendered, raster)

        def render_player_missile(i, raster_in):
            render_result = self.jr.render_at_clipped(raster_in, state.player_missile_x[i], state.player_missile_y[i], missile_mask)
            return jnp.where(state.player_missile_active[i] == 1, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.NUM_PLAYER_MISSILES, render_player_missile, raster)

        def render_enemy_missile(i, raster_in):
            render_result = self.jr.render_at_clipped(raster_in, state.enemy_missile_x[i], state.enemy_missile_y[i], missile_mask)
            return jnp.where(state.enemy_missile_active[i] == 1, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.NUM_ENEMY_MISSILES, render_enemy_missile, raster)

        raster = self.jr.render_at(raster, 0, AirRaidConstants.HEIGHT - 20, self.SHAPE_MASKS["black_bar"])

        score_value = state.score
        score_digits = self.jr.int_to_digits(score_value, max_digits=6)
        is_score_zero = score_value == 0
        significant_mask = score_digits > 0
        indices = jnp.arange(6, dtype=jnp.int32)
        first_significant_idx = jnp.min(jnp.where(significant_mask, indices, 6))
        start_index = jax.lax.select(is_score_zero, 5, first_significant_idx)
        num_to_render = jax.lax.select(is_score_zero, 1, 6 - first_significant_idx)

        raster = self.jr.render_label_selective(
            raster,
            30,
            5,
            score_digits,
            score_digit_masks,
            start_index,
            num_to_render,
            spacing=self.score_digit_spacing,
            max_digits_to_render=6,
        )

        lives = state.player_lives

        def render_life(i, raster_in):
            icon_x = 30 + i * self.life_spacing
            render_result = self.jr.render_at(raster_in, icon_x, AirRaidConstants.HEIGHT - 17, life_mask)
            return jnp.where(i < lives - 1, render_result, raster_in)

        raster = jax.lax.fori_loop(0, 2, render_life, raster)

        rgb_frame = self.jr.render_from_palette(raster, self.PALETTE)

        should_flash = state.flash_counter > 0
        flash_on = (state.flash_counter % 10) < 5
        white_raster = jnp.full_like(rgb_frame, 255)
        flash_mask = jnp.logical_and(should_flash, flash_on)
        rgb_frame = jnp.where(flash_mask[..., None], white_raster, rgb_frame)

        return rgb_frame
