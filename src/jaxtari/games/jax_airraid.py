import os
from functools import partial
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import chex
from flax import struct
import jaxtari.spaces as spaces

from jaxtari.rendering import jax_rendering_utils as render_utils
from jaxtari.environment import JaxEnvironment, JaxtariAction as Action, ObjectObservation
from jaxtari.renderers import JAXGameRenderer


def _decode_bitmap(hex_rows: str, width: int, row_repeat: int = 1, col_repeat: int = 1) -> np.ndarray:
    """Decodes whitespace-separated hex row values (MSB = leftmost pixel) into a boolean bitmap."""
    rows = [[(int(token, 16) >> (width - 1 - c)) & 1 for c in range(width)] for token in hex_rows.split()]
    bitmap = np.array(rows, dtype=bool)
    return np.repeat(np.repeat(bitmap, row_repeat, axis=0), col_repeat, axis=1)


# Building damage stages as extracted from ALE. Each building is a 8x17 grid of 4x2 pixel cells
# (one hex byte per cell row). Stages 0-8 are hit stages, 9-12 the collapse animation, 13 the rubble.
BUILDING_STAGES = (
    "00 42 ff db ff db ff db ff db ff db ff db ff db ff",
    "00 00 b7 db ff db ff db ff db ff db ff db ff db ff",
    "00 00 19 db ff db ff db ff db ff db ff db ff db ff",
    "00 00 01 11 3f db ff db ff db ff db ff db ff db ff",
    "00 00 00 10 39 5b ff db ff db ff db ff db ff db ff",
    "00 00 00 00 00 48 fb db ff db ff db ff db ff db ff",
    "00 00 00 00 00 00 40 c9 fb db ff db ff db ff db ff",
    "00 00 00 00 00 00 00 08 ba db ff db ff db ff db ff",
    "00 00 00 00 00 00 00 00 00 02 7e da fe db ff db ff",
    "00 00 00 00 00 00 00 20 80 42 21 14 d5 28 bf db ff",
    "00 00 00 00 00 41 46 d0 13 c4 04 52 58 18 bb dd 42",
    "00 00 00 00 81 82 40 c2 51 2a c9 82 46 2b 04 52 93",
    "00 00 00 00 81 82 40 c2 51 2a c9 82 46 2b 04 52 93",
    "00 00 00 00 00 00 00 00 00 00 00 00 00 20 21 fd 42",
)
_BUILDING_TOP_Y = 176
_BUILDING_CELLS = np.stack([_decode_bitmap(stage, 8) for stage in BUILDING_STAGES])  # (14, 17, 8)
# Topmost occupied pixel row per 4px building column (ground level if the column is empty)
_BUILDING_COLUMN_TOPS = np.where(
    _BUILDING_CELLS.any(axis=1),
    _BUILDING_TOP_Y + 2 * np.argmax(_BUILDING_CELLS, axis=1),
    _BUILDING_TOP_Y + 2 * _BUILDING_CELLS.shape[1],
).astype(np.int32)  # (14, 8)


class AirRaidConstants(struct.PyTreeNode):
    # Game environment
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=250)

    # Player
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=14)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=12)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)
    PLAYER_INITIAL_X: int = struct.field(pytree_node=False, default=61)
    PLAYER_INITIAL_Y: int = struct.field(pytree_node=False, default=156)
    PLAYER_MIN_X: int = struct.field(pytree_node=False, default=21)
    PLAYER_MAX_X: int = struct.field(pytree_node=False, default=128)
    MAX_PLAYER_LIVES: int = struct.field(pytree_node=False, default=4)
    PLAYER_HIT_FRAMES: int = struct.field(pytree_node=False, default=4)  # Hit sprite before the explosion
    PLAYER_DEATH_FRAMES: int = struct.field(pytree_node=False, default=20)  # Hit sprite + explosion animation

    # Buildings
    NUM_BUILDINGS: int = struct.field(pytree_node=False, default=2)
    BUILDING_WIDTH: int = struct.field(pytree_node=False, default=32)
    BUILDING_HEIGHT: int = struct.field(pytree_node=False, default=34)
    BUILDING_COLLAPSE_DAMAGE: int = struct.field(pytree_node=False, default=9)  # 9th hit starts the collapse
    MAX_BUILDING_DAMAGE: int = struct.field(pytree_node=False, default=13)  # Rubble, building destroyed
    BUILDING_COLLAPSE_FRAMES: int = struct.field(pytree_node=False, default=30)  # Frames per collapse stage
    BUILDING_INITIAL_X: int = struct.field(pytree_node=False, default=34)
    BUILDING_INITIAL_Y: int = struct.field(pytree_node=False, default=_BUILDING_TOP_Y)
    BUILDING_SCROLL_INTERVAL: int = struct.field(pytree_node=False, default=2)  # Scrolls 1px every 2 frames
    BUILDING_SPACING: int = struct.field(pytree_node=False, default=80)
    # Enemy bombs hit a building a few pixels before visually touching it (as in ALE)
    BUILDING_HIT_LOOKAHEAD: int = struct.field(pytree_node=False, default=18)

    # Per damage stage: topmost pixel row of each 4px building column, plus the overall top and height
    BUILDING_COLUMN_TOPS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(_BUILDING_COLUMN_TOPS),
    )
    BUILDING_Y_POSITIONS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(_BUILDING_COLUMN_TOPS.min(axis=1)),
    )
    BUILDING_HEIGHTS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(_BUILDING_TOP_Y + 34 - _BUILDING_COLUMN_TOPS.min(axis=1)),
    )

    # HUD
    SCORE_X: int = struct.field(pytree_node=False, default=96)  # Left edge of the least significant digit
    SCORE_Y: int = struct.field(pytree_node=False, default=11)
    LIFE_X: int = struct.field(pytree_node=False, default=55)
    LIFE_Y: int = struct.field(pytree_node=False, default=219)
    LIFE_SPACING: int = struct.field(pytree_node=False, default=8)
    HUD_Y: int = struct.field(pytree_node=False, default=210)
    HUD_FLASH_FRAMES: int = struct.field(pytree_node=False, default=13)  # HUD flashes when a building is hit

    # Enemies: one enemy per lane (left, middle, right), types 0=25, 1=50, 2=75, 3=100 points
    TOTAL_ENEMIES: int = struct.field(pytree_node=False, default=3)
    ENEMY_SPAWN_Y: int = struct.field(pytree_node=False, default=35)
    ENEMY_BOTTOM_Y: int = struct.field(pytree_node=False, default=149)  # Enemies vanish below the horizon here
    ENEMY_HORIZON_Y: int = struct.field(pytree_node=False, default=152)  # Enemy sprites are clipped below this row
    ENEMY_SLOW_INTERVAL: int = struct.field(pytree_node=False, default=3)  # 50/100 enemies move 1px every 3 frames
    ENEMY_RESPAWN_DELAY: int = struct.field(pytree_node=False, default=33)  # After passing the horizon
    ENEMY_KILL_RESPAWN_DELAY: int = struct.field(pytree_node=False, default=69)  # After being shot
    ENEMY_EXPLOSION_FRAMES: int = struct.field(pytree_node=False, default=9)
    ENEMY_WAVE_DELAY: int = struct.field(pytree_node=False, default=30)  # After the player respawns
    ENEMY_FIRE_MAX_Y: int = struct.field(pytree_node=False, default=94)  # Enemies only drop bombs high up
    ENEMY_FIRE_PROB: float = struct.field(pytree_node=False, default=0.01)
    ENEMY_FIRE_COOLDOWN: int = struct.field(pytree_node=False, default=17)
    ENEMY_BUILDING_HIT_COOLDOWN: int = struct.field(pytree_node=False, default=180)
    ENEMY_START_COOLDOWN: int = struct.field(pytree_node=False, default=150)  # No bombs right after the start
    ENEMY_HITBOX_OFFSET: int = struct.field(pytree_node=False, default=5)
    ENEMY_HITBOX_WIDTH: int = struct.field(pytree_node=False, default=7)
    ENEMY_INITIAL_X: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([26, 74, 124], dtype=jnp.int32)
    )
    ENEMY_INITIAL_TYPES: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1, 3, 1], dtype=jnp.int32)
    )
    ENEMY_LANE_MIN_X: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([23, 53, 119], dtype=jnp.int32)
    )
    ENEMY_LANE_MAX_X: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([28, 78, 126], dtype=jnp.int32)
    )
    ENEMY_WIDTHS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([16, 14, 14, 14], dtype=jnp.int32)
    )
    ENEMY_HEIGHTS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([18, 16, 16, 14], dtype=jnp.int32)
    )

    # Missiles
    MISSILE_WIDTH: int = struct.field(pytree_node=False, default=2)
    MISSILE_HEIGHT: int = struct.field(pytree_node=False, default=2)
    NUM_PLAYER_MISSILES: int = struct.field(pytree_node=False, default=1)
    NUM_ENEMY_MISSILES: int = struct.field(pytree_node=False, default=1)
    PLAYER_MISSILE_SPEED: int = struct.field(pytree_node=False, default=-3)
    PLAYER_MISSILE_MIN_Y: int = struct.field(pytree_node=False, default=31)
    ENEMY_MISSILE_SPEED: int = struct.field(pytree_node=False, default=3)
    ENEMY_MISSILE_MAX_Y: int = struct.field(pytree_node=False, default=193)  # Bombs hit the ground here


DEFAULT_AIRRAID_CONSTANTS = AirRaidConstants()

# Immutable state container
class AirRaidState(struct.PyTreeNode):
    player_x: chex.Array = struct.field()
    player_y: chex.Array = struct.field()
    player_lives: chex.Array = struct.field()
    player_visible: chex.Array = struct.field()
    player_death_timer: chex.Array = struct.field()  # Frames since the player was hit (0 when alive)

    building_x: chex.Array = struct.field()
    building_y: chex.Array = struct.field()
    building_damage: chex.Array = struct.field()
    building_timer: chex.Array = struct.field()  # Frames spent in the current collapse stage

    enemy_x: chex.Array = struct.field()
    enemy_y: chex.Array = struct.field()
    enemy_type: chex.Array = struct.field()
    enemy_active: chex.Array = struct.field()
    enemy_timer: chex.Array = struct.field()  # Frames until an inactive enemy respawns

    player_missile_x: chex.Array = struct.field()
    player_missile_y: chex.Array = struct.field()
    player_missile_active: chex.Array = struct.field()

    enemy_missile_x: chex.Array = struct.field()
    enemy_missile_y: chex.Array = struct.field()
    enemy_missile_active: chex.Array = struct.field()
    enemy_missile_cooldown: chex.Array = struct.field()  # Frames until enemies may drop the next bomb

    score: chex.Array = struct.field()
    step_counter: chex.Array = struct.field()
    flash_counter: chex.Array = struct.field()  # Remaining frames of the HUD flash after a building was hit
    # ALE shows the frame in greyscale while the enemy bomb slot is in use (PAL color loss)
    color_loss: chex.Array = struct.field()
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
    # Check if left or right button was pressed
    move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
    move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

    player_x = jnp.where(
        move_left,
        jnp.maximum(player_x - AirRaidConstants.PLAYER_SPEED, AirRaidConstants.PLAYER_MIN_X),
        player_x
    )

    player_x = jnp.where(
        move_right,
        jnp.minimum(player_x + AirRaidConstants.PLAYER_SPEED, AirRaidConstants.PLAYER_MAX_X),
        player_x
    )

    return player_x

@jax.jit
def spawn_enemy(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Counts down the respawn timers of inactive enemies and respawns them at the top of their lane.
    Enemies only respawn while the player is alive. Enemies that left through the bottom keep their
    x position, enemies that were shot down (or start a new wave) get a new position in their lane.
    Args: state: Current game state
    Returns: Updated enemy x, y, active flags, timers and RNG
    """
    consts = DEFAULT_AIRRAID_CONSTANTS
    rng, pos_key = random.split(state.rng)

    inactive = state.enemy_active == 0
    enemy_timer = jnp.where(inactive, jnp.maximum(state.enemy_timer - 1, 0), state.enemy_timer)
    should_spawn = jnp.logical_and(
        jnp.logical_and(inactive, enemy_timer == 0),
        state.player_visible == 1,
    )

    lane_x = random.randint(
        pos_key,
        shape=(AirRaidConstants.TOTAL_ENEMIES,),
        minval=consts.ENEMY_LANE_MIN_X,
        maxval=consts.ENEMY_LANE_MAX_X + 1,
    )

    relocate = state.enemy_y < AirRaidConstants.ENEMY_BOTTOM_Y
    enemy_x = jnp.where(jnp.logical_and(should_spawn, relocate), lane_x, state.enemy_x)
    enemy_y = jnp.where(should_spawn, jnp.int32(AirRaidConstants.ENEMY_SPAWN_Y), state.enemy_y)
    enemy_active = jnp.where(should_spawn, jnp.int32(1), state.enemy_active)

    return enemy_x, enemy_y, enemy_active, enemy_timer, rng

@jax.jit
def update_enemies(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Updates all enemy positions. Enemies move straight down their lane."""
    # 25 and 75 point enemies descend 1px per frame, 50 and 100 point enemies 1px every 3 frames.
    # While the player is dead, all enemies hurry down so that the next wave can start.
    is_fast = jnp.logical_or(state.enemy_type == 0, state.enemy_type == 2)
    is_fast = jnp.logical_or(is_fast, state.player_visible == 0)
    slow_tick = state.step_counter % AirRaidConstants.ENEMY_SLOW_INTERVAL == 0
    moves = jnp.logical_and(state.enemy_active == 1, jnp.logical_or(is_fast, slow_tick))
    enemy_y = jnp.where(moves, state.enemy_y + 1, state.enemy_y)

    # Enemies that pass the horizon vanish and come back as the same type after a short delay
    reached_bottom = jnp.logical_and(state.enemy_active == 1, enemy_y >= AirRaidConstants.ENEMY_BOTTOM_Y)
    enemy_active = jnp.where(reached_bottom, jnp.int32(0), state.enemy_active)
    enemy_timer = jnp.where(reached_bottom, jnp.int32(AirRaidConstants.ENEMY_RESPAWN_DELAY), state.enemy_timer)

    return enemy_y, enemy_active, enemy_timer


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
    # The missile leaves from the middle of the ship (it moves once before it is first drawn)
    missile_y = state.player_y + 4 - AirRaidConstants.PLAYER_MISSILE_SPEED

    # Update missile state if firing
    player_missile_x = state.player_missile_x.at[first_inactive].set(
        jnp.where(should_fire, missile_x, state.player_missile_x[first_inactive])
    )
    player_missile_y = state.player_missile_y.at[first_inactive].set(
        jnp.where(should_fire, missile_y, state.player_missile_y[first_inactive])
    )
    player_missile_active = state.player_missile_active.at[first_inactive].set(
        jnp.where(should_fire, 1, state.player_missile_active[first_inactive])
    )

    return player_missile_x, player_missile_y, player_missile_active

@jax.jit
def fire_enemy_missiles(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Lets the enemy closest to the player drop a bomb.

    Args:
        state: Current game state

    Returns:
        Updated enemy missile positions, active flags and RNG
    """
    consts = DEFAULT_AIRRAID_CONSTANTS
    rng = state.rng
    enemy_missile_x = state.enemy_missile_x
    enemy_missile_y = state.enemy_missile_y
    enemy_missile_active = state.enemy_missile_active

    # Find the first inactive missile
    inactive_missile_mask = 1 - enemy_missile_active
    inactive_indices = jnp.where(inactive_missile_mask, jnp.arange(AirRaidConstants.NUM_ENEMY_MISSILES), -1)
    first_inactive = jnp.max(inactive_indices)  # Get the highest valid index

    rng, fire_key = random.split(rng)
    fire_prob = random.uniform(fire_key)

    # The enemy horizontally closest to the player is the one that drops the bomb
    enemy_widths = consts.ENEMY_WIDTHS[state.enemy_type]
    enemy_centers = state.enemy_x + enemy_widths // 2
    player_center = state.player_x + AirRaidConstants.PLAYER_WIDTH // 2
    distances = jnp.where(state.enemy_active == 1, jnp.abs(enemy_centers - player_center), jnp.iinfo(jnp.int32).max)
    firing_enemy_idx = jnp.argmin(distances)

    selected_enemy_active = state.enemy_active[firing_enemy_idx] == 1
    high_enough = state.enemy_y[firing_enemy_idx] <= AirRaidConstants.ENEMY_FIRE_MAX_Y
    no_active_missiles = jnp.sum(enemy_missile_active) == 0
    cooled_down = state.enemy_missile_cooldown == 0
    player_targetable = jnp.logical_and(state.player_visible == 1, state.player_lives > 0)

    can_fire = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(fire_prob < AirRaidConstants.ENEMY_FIRE_PROB, first_inactive >= 0),
            jnp.logical_and(selected_enemy_active, high_enough),
        ),
        jnp.logical_and(jnp.logical_and(no_active_missiles, cooled_down), player_targetable)
    )

    # Bombs are dropped from below the enemy (they move once before they are first drawn)
    enemy_missile_x = enemy_missile_x.at[first_inactive].set(
        jnp.where(
            can_fire,
            enemy_centers[firing_enemy_idx],
            enemy_missile_x[first_inactive]
        )
    )

    enemy_missile_y = enemy_missile_y.at[first_inactive].set(
        jnp.where(
            can_fire,
            state.enemy_y[firing_enemy_idx] + 15 - AirRaidConstants.ENEMY_MISSILE_SPEED,
            enemy_missile_y[first_inactive]
        )
    )

    enemy_missile_active = enemy_missile_active.at[first_inactive].set(
        jnp.where(can_fire, 1, enemy_missile_active[first_inactive])
    )

    return enemy_missile_x, enemy_missile_y, enemy_missile_active, rng

@jax.jit
def update_missiles(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Updates the positions of all missiles and deactivates those that leave the playfield.

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

    # Player missiles vanish at the top of the playfield, enemy bombs when they hit the ground
    player_missile_active = jnp.where(
        player_missile_y < AirRaidConstants.PLAYER_MISSILE_MIN_Y,
        0,
        state.player_missile_active
    )

    enemy_missile_active = jnp.where(
        enemy_missile_y > AirRaidConstants.ENEMY_MISSILE_MAX_Y,
        0,
        state.enemy_missile_active
    )

    return player_missile_y, player_missile_active, enemy_missile_y, enemy_missile_active

@jax.jit
def detect_collisions(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Detects all collisions between game objects."""
    consts = DEFAULT_AIRRAID_CONSTANTS
    enemy_active = state.enemy_active
    player_missile_active = state.player_missile_active
    enemy_missile_active = state.enemy_missile_active
    score = state.score
    player_lives = state.player_lives
    building_damage = state.building_damage

    score_values = jnp.array([25, 50, 75, 100], dtype=jnp.int32)
    enemy_sprite_widths = consts.ENEMY_WIDTHS[state.enemy_type]
    enemy_hitbox_x = state.enemy_x + (enemy_sprite_widths - 14) // 2 + AirRaidConstants.ENEMY_HITBOX_OFFSET
    enemy_heights = consts.ENEMY_HEIGHTS[state.enemy_type]

    def process_player_missile(carry, pm):
        carry_enemy_active, carry_player_missile_active, carry_score = carry
        is_missile_active = carry_player_missile_active[pm] == 1
        missile_x = state.player_missile_x[pm]
        missile_y = state.player_missile_y[pm]

        def enemy_collision_fn(hx, ey, eh, ea):
            collision = jnp.logical_and(
                jnp.logical_and(
                    missile_x < hx + AirRaidConstants.ENEMY_HITBOX_WIDTH,
                    missile_x + AirRaidConstants.MISSILE_WIDTH > hx,
                ),
                jnp.logical_and(
                    missile_y < ey + eh,
                    missile_y + AirRaidConstants.MISSILE_HEIGHT > ey,
                ),
            )
            return jnp.logical_and(jnp.logical_and(collision, is_missile_active), ea == 1)

        effective_collisions = jax.vmap(enemy_collision_fn)(
            enemy_hitbox_x,
            state.enemy_y,
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
        carry_enemy_missile_active, carry_player_lives, carry_building_damage, carry_building_hit = carry
        is_missile_active = carry_enemy_missile_active[em] == 1
        missile_x = state.enemy_missile_x[em]
        missile_y = state.enemy_missile_y[em]

        def building_collision_fn(bx, damage):
            # Buildings wrap around the screen, so measure the missile position relative to the building
            rel_left = (missile_x - bx) % AirRaidConstants.WIDTH
            rel_right = (missile_x + AirRaidConstants.MISSILE_WIDTH - 1 - bx) % AirRaidConstants.WIDTH
            left_in = rel_left < AirRaidConstants.BUILDING_WIDTH
            right_in = rel_right < AirRaidConstants.BUILDING_WIDTH
            column_tops = consts.BUILDING_COLUMN_TOPS[damage]
            ground = AirRaidConstants.BUILDING_INITIAL_Y + AirRaidConstants.BUILDING_HEIGHT
            left_top = jnp.where(left_in, column_tops[jnp.clip(rel_left // 4, 0, 7)], ground)
            right_top = jnp.where(right_in, column_tops[jnp.clip(rel_right // 4, 0, 7)], ground)
            top = jnp.minimum(left_top, right_top)
            collision = jnp.logical_and(
                missile_y + AirRaidConstants.BUILDING_HIT_LOOKAHEAD >= top,
                damage < AirRaidConstants.BUILDING_COLLAPSE_DAMAGE,
            )
            return jnp.logical_and(collision, is_missile_active)

        effective_building_collisions = jax.vmap(building_collision_fn)(state.building_x, carry_building_damage)
        # A bomb can only damage one building
        effective_building_collisions = jnp.logical_and(
            effective_building_collisions,
            jnp.cumsum(effective_building_collisions) == 1,
        )
        carry_building_damage = carry_building_damage + effective_building_collisions.astype(carry_building_damage.dtype)

        # Bombs hit the ship one step before overlapping it (as in ALE)
        player_collision = jnp.logical_and(
            jnp.logical_and(
                missile_x < state.player_x + AirRaidConstants.PLAYER_WIDTH,
                missile_x + AirRaidConstants.MISSILE_WIDTH > state.player_x,
            ),
            jnp.logical_and(
                missile_y < state.player_y + AirRaidConstants.PLAYER_HEIGHT,
                missile_y + AirRaidConstants.MISSILE_HEIGHT + AirRaidConstants.ENEMY_MISSILE_SPEED > state.player_y,
            ),
        )
        effective_player_collision = jnp.logical_and(player_collision, is_missile_active)
        effective_player_collision = jnp.logical_and(effective_player_collision, state.player_visible == 1)

        building_hit = jnp.any(effective_building_collisions)
        missile_deactivated = jnp.logical_or(building_hit, effective_player_collision)
        carry_enemy_missile_active = carry_enemy_missile_active.at[em].set(
            jnp.where(missile_deactivated, 0, carry_enemy_missile_active[em])
        )
        carry_player_lives = jnp.where(effective_player_collision, carry_player_lives - 1, carry_player_lives)
        carry_building_hit = jnp.logical_or(carry_building_hit, building_hit)

        return (carry_enemy_missile_active, carry_player_lives, carry_building_damage, carry_building_hit), None

    (enemy_missile_active, player_lives, building_damage, building_hit), _ = jax.lax.scan(
        process_enemy_missile,
        (enemy_missile_active, player_lives, building_damage, jnp.array(False)),
        jnp.arange(AirRaidConstants.NUM_ENEMY_MISSILES),
    )

    return enemy_active, player_missile_active, enemy_missile_active, score, player_lives, building_damage, building_hit


class JaxAirRaid(JaxEnvironment[AirRaidState, AirRaidObservation, AirRaidInfo, AirRaidConstants]):
    # Minimal ALE action set for Air Raid:
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
        consts = DEFAULT_AIRRAID_CONSTANTS
        # Initialize building positions to match ALE (two houses on the ground line)
        building_x = jnp.array([
            AirRaidConstants.BUILDING_INITIAL_X,
            AirRaidConstants.BUILDING_INITIAL_X + AirRaidConstants.BUILDING_SPACING,
        ])
        building_y = jnp.array([AirRaidConstants.BUILDING_INITIAL_Y, AirRaidConstants.BUILDING_INITIAL_Y])
        building_damage = jnp.zeros(AirRaidConstants.NUM_BUILDINGS, dtype=jnp.int32)
        building_timer = jnp.zeros(AirRaidConstants.NUM_BUILDINGS, dtype=jnp.int32)

        # The first wave always starts with the same formation (left-to-right: 50, 100, 50 points)
        enemy_x = consts.ENEMY_INITIAL_X
        enemy_y = jnp.full(AirRaidConstants.TOTAL_ENEMIES, AirRaidConstants.ENEMY_SPAWN_Y, dtype=jnp.int32)
        enemy_type = consts.ENEMY_INITIAL_TYPES
        enemy_active = jnp.ones(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_timer = jnp.zeros(AirRaidConstants.TOTAL_ENEMIES, dtype=jnp.int32)

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
            player_death_timer=jnp.array(0),
            building_x=building_x,
            building_y=building_y,
            building_damage=building_damage,
            building_timer=building_timer,
            enemy_x=enemy_x,
            enemy_y=enemy_y,
            enemy_type=enemy_type,
            enemy_active=enemy_active,
            enemy_timer=enemy_timer,
            player_missile_x=player_missile_x,
            player_missile_y=player_missile_y,
            player_missile_active=player_missile_active,
            enemy_missile_x=enemy_missile_x,
            enemy_missile_y=enemy_missile_y,
            enemy_missile_active=enemy_missile_active,
            enemy_missile_cooldown=jnp.array(AirRaidConstants.ENEMY_START_COOLDOWN),
            score=jnp.array(0),
            step_counter=jnp.array(0),
            flash_counter=jnp.array(0),
            color_loss=jnp.array(0),
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

        # Buildings scroll to the right by 1px every other frame and wrap around the screen
        scroll = (state.step_counter % AirRaidConstants.BUILDING_SCROLL_INTERVAL == 1).astype(jnp.int32)
        new_building_x = (state.building_x + scroll) % AirRaidConstants.WIDTH

        # Buildings that received their final hit crumble stage by stage until only rubble remains
        collapsing = jnp.logical_and(
            state.building_damage >= AirRaidConstants.BUILDING_COLLAPSE_DAMAGE,
            state.building_damage < AirRaidConstants.MAX_BUILDING_DAMAGE,
        )
        building_timer = jnp.where(collapsing, state.building_timer + 1, 0)
        collapse_advance = jnp.logical_and(collapsing, building_timer >= AirRaidConstants.BUILDING_COLLAPSE_FRAMES)
        building_damage = state.building_damage + collapse_advance.astype(jnp.int32)
        building_timer = jnp.where(collapse_advance, 0, building_timer)

        player_is_visible = state.player_visible == 1

        # Update player position
        stepped_player_x = player_step(state.player_x, action)
        new_player_x = jnp.where(player_is_visible, stepped_player_x, state.player_x)

        noop_action = jnp.array(Action.NOOP, dtype=action.dtype)
        effective_action = jnp.where(player_is_visible, action, noop_action)

        # Respawn enemies whose timers ran out (only while the player is alive)
        new_enemy_x, new_enemy_y, new_enemy_active, new_enemy_timer, new_rng = spawn_enemy(
            state.replace(player_x=new_player_x)
        )

        # Move enemies down their lanes
        updated_enemy_y, updated_enemy_active, updated_enemy_timer = update_enemies(
            state.replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_active=new_enemy_active,
                enemy_timer=new_enemy_timer,
            )
        )

        # Handle player firing missiles
        new_player_missile_x, new_player_missile_y, new_player_missile_active = fire_player_missile(
            state.replace(player_x=new_player_x),
            effective_action
        )

        # Handle enemy firing missiles
        new_enemy_missile_x, new_enemy_missile_y, new_enemy_missile_active, newer_rng = fire_enemy_missiles(
            state.replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_active=updated_enemy_active,
                rng=new_rng
            )
        )

        # Update missile positions
        updated_player_missile_y, updated_player_missile_active, updated_enemy_missile_y, updated_enemy_missile_active = update_missiles(
            state.replace(
                player_missile_y=new_player_missile_y,
                player_missile_active=new_player_missile_active,
                enemy_missile_y=new_enemy_missile_y,
                enemy_missile_active=new_enemy_missile_active,
            )
        )

        # Detect and handle collisions
        final_enemy_active, final_player_missile_active, final_enemy_missile_active, new_score, new_player_lives, final_building_damage, building_hit = detect_collisions(
            state.replace(
                player_x=new_player_x,
                building_x=new_building_x,
                building_damage=building_damage,
                enemy_x=new_enemy_x,
                enemy_y=updated_enemy_y,
                enemy_active=updated_enemy_active,
                player_missile_x=new_player_missile_x,
                player_missile_y=updated_player_missile_y,
                player_missile_active=updated_player_missile_active,
                enemy_missile_x=new_enemy_missile_x,
                enemy_missile_y=updated_enemy_missile_y,
                enemy_missile_active=updated_enemy_missile_active
            )
        )

        # Shot down enemies explode and return later as a random type
        newer_rng, type_key = random.split(newer_rng)
        random_types = random.randint(type_key, shape=(AirRaidConstants.TOTAL_ENEMIES,), minval=0, maxval=4)
        enemy_killed = jnp.logical_and(updated_enemy_active == 1, final_enemy_active == 0)
        final_enemy_type = jnp.where(enemy_killed, random_types, state.enemy_type)
        final_enemy_timer = jnp.where(
            enemy_killed, jnp.int32(AirRaidConstants.ENEMY_KILL_RESPAWN_DELAY), updated_enemy_timer
        )

        # After a bomb is gone the enemies wait a moment before dropping the next one
        enemy_missile_ended = jnp.logical_and(
            jnp.any(new_enemy_missile_active == 1), jnp.all(final_enemy_missile_active == 0)
        )
        enemy_missile_cooldown = jnp.where(
            building_hit,
            AirRaidConstants.ENEMY_BUILDING_HIT_COOLDOWN,
            jnp.where(
                enemy_missile_ended,
                AirRaidConstants.ENEMY_FIRE_COOLDOWN,
                jnp.maximum(state.enemy_missile_cooldown - 1, 0),
            ),
        )

        previous_life_milestone = state.score // 1000
        new_life_milestone = new_score // 1000
        bonus_lives = jnp.maximum(new_life_milestone - previous_life_milestone, 0)
        adjusted_player_lives = jnp.minimum(new_player_lives + bonus_lives, AirRaidConstants.MAX_PLAYER_LIVES)

        life_lost = new_player_lives < state.player_lives
        enemies_remaining = jnp.any(final_enemy_active == 1)
        player_is_alive = adjusted_player_lives > 0

        # After being hit the ship explodes; the next ship only appears once all enemies have left the screen
        player_death_timer = jnp.where(
            life_lost,
            jnp.int32(1),
            jnp.where(state.player_death_timer > 0, state.player_death_timer + 1, 0),
        )
        death_animation_done = player_death_timer > AirRaidConstants.PLAYER_DEATH_FRAMES
        waiting_for_respawn = jnp.logical_or(state.player_visible == 0, life_lost)
        can_reappear = jnp.logical_and(jnp.logical_not(enemies_remaining), death_animation_done)

        next_player_visible = jnp.where(
            player_is_alive,
            jnp.where(jnp.logical_and(waiting_for_respawn, jnp.logical_not(can_reappear)), jnp.int32(0), jnp.int32(1)),
            jnp.int32(0),
        )

        player_reappears = jnp.logical_and(state.player_visible == 0, next_player_visible == 1)
        final_player_x = jnp.where(player_reappears, jnp.int32(AirRaidConstants.PLAYER_INITIAL_X), new_player_x)
        final_player_y = jnp.where(player_reappears, jnp.int32(AirRaidConstants.PLAYER_INITIAL_Y), state.player_y)
        player_death_timer = jnp.where(player_reappears, 0, player_death_timer)

        # A new wave of random enemies at new positions starts shortly after the next ship appeared
        final_enemy_timer = jnp.where(
            player_reappears, jnp.int32(AirRaidConstants.ENEMY_WAVE_DELAY), final_enemy_timer
        )
        final_enemy_type = jnp.where(player_reappears, random_types, final_enemy_type)
        final_enemy_y = jnp.where(player_reappears, jnp.int32(AirRaidConstants.ENEMY_SPAWN_Y), updated_enemy_y)

        final_player_missile_active = jnp.where(
            life_lost,
            jnp.zeros_like(final_player_missile_active),
            final_player_missile_active,
        )

        # The HUD flashes whenever a building is hit
        flash_counter = jnp.where(
            building_hit,
            AirRaidConstants.HUD_FLASH_FRAMES,
            jnp.maximum(state.flash_counter - 1, 0),
        )

        # The frame turns grey while the bomb slot was in use on the previous frame: a bomb was falling or
        # the slot is still blocked after a building hit / at the start. The color additionally flips for
        # one frame when the player's missile resets at the top of the screen.
        bomb_slot_in_use = jnp.logical_or(
            jnp.any(state.enemy_missile_active == 1),
            state.enemy_missile_cooldown > AirRaidConstants.ENEMY_FIRE_COOLDOWN,
        )
        player_missile_reached_top = jnp.any(
            jnp.logical_and(
                jnp.logical_and(new_player_missile_active == 1, updated_player_missile_active == 0),
                updated_player_missile_y < AirRaidConstants.PLAYER_MISSILE_MIN_Y,
            )
        )
        color_loss = jnp.logical_xor(bomb_slot_in_use, player_missile_reached_top).astype(jnp.int32)

        new_state = state.replace(
            player_x=final_player_x,
            player_y=final_player_y,
            player_lives=adjusted_player_lives,
            player_visible=next_player_visible,
            player_death_timer=player_death_timer,
            building_x=new_building_x,
            building_y=state.building_y,
            building_damage=final_building_damage,
            building_timer=building_timer,
            enemy_x=new_enemy_x,
            enemy_y=final_enemy_y,
            enemy_type=final_enemy_type,
            enemy_active=final_enemy_active,
            enemy_timer=final_enemy_timer,
            player_missile_x=new_player_missile_x,
            player_missile_y=updated_player_missile_y,
            player_missile_active=final_player_missile_active,
            enemy_missile_x=new_enemy_missile_x,
            enemy_missile_y=updated_enemy_missile_y,
            enemy_missile_active=final_enemy_missile_active,
            enemy_missile_cooldown=enemy_missile_cooldown,
            score=new_score,
            step_counter=state.step_counter + 1,
            flash_counter=flash_counter,
            color_loss=color_loss,
            rng=newer_rng,
        )

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
        consts = DEFAULT_AIRRAID_CONSTANTS
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
        building_y = consts.BUILDING_Y_POSITIONS[state.building_damage]
        building_h = consts.BUILDING_HEIGHTS[state.building_damage]
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

        enemy_widths = consts.ENEMY_WIDTHS[state.enemy_type]
        enemy_heights = consts.ENEMY_HEIGHTS[state.enemy_type]
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
        """Check if game over conditions are met (ignoring the end-of-game animations)"""
        # Game is over if player has no lives left
        player_dead = jnp.less_equal(state.player_lives, 0)

        # Game is over if both buildings are completely destroyed
        buildings_destroyed = jnp.all(state.building_damage >= AirRaidConstants.MAX_BUILDING_DAMAGE)

        return jnp.logical_or(player_dead, buildings_destroyed)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AirRaidState) -> bool:
        # After the last ship is lost, ALE ends the game once the remaining enemies have left the screen
        last_ship_gone = jnp.logical_and(
            state.player_death_timer > AirRaidConstants.PLAYER_DEATH_FRAMES,
            jnp.logical_not(jnp.any(state.enemy_active == 1)),
        )
        player_finished = jnp.logical_and(jnp.less_equal(state.player_lives, 0), last_ship_gone)
        buildings_destroyed = jnp.all(state.building_damage >= AirRaidConstants.MAX_BUILDING_DAMAGE)
        return jnp.logical_or(player_finished, buildings_destroyed)


class AirRaidRenderer(JAXGameRenderer):
    # Bitmaps extracted from ALE (hex rows, MSB = leftmost pixel)
    PLAYER_FRAMES = ("00c0 03f0 3333 30c3 03f0 0c0c", "00c0 03f0 3ccc 30c3 03f0 0c0c")
    PLAYER_DEATH_FRAMES = (
        "03c0 0ff0 3ffc 0ff0 03c0",  # Hit
        "c003 0c30 03c0 0c30 c003",  # Explosion
        "0c30 c003 300c c003 0c30",
    )
    ENEMY_EXPLOSION_FRAMES = (
        "c003 3c3c 3ffc 0ff0 0ff0 3ffc 3c3c c003",
        "c003 0c30 300c 03c0 03c0 300c 0c30 c003",
        "0c30 c003 300c c003 c003 300c c003 0c30",
    )
    SCORE_DIGITS = (
        "1e 39 39 39 39 39 39 1e", "1c 0c 0c 0c 0c 0c 0c 0c", "1e 27 07 07 1e 20 23 3f",
        "1e 27 07 0e 0e 07 27 1f", "26 26 26 26 26 3f 06 06", "3f 20 20 3e 07 07 27 3e",
        "1e 21 20 3e 27 27 27 1e", "3f 23 03 03 06 06 0c 0c", "1e 39 39 1e 1e 27 27 1e",
        "1e 39 39 39 1f 01 21 1e",
    )

    # Colors as shown by ALE in color frames
    SKY_COLOR = (112, 0, 92)
    PLAYER_ROW_COLORS = ((121, 181, 236), (87, 139, 201), (47, 90, 160), (87, 139, 201), (121, 181, 236), (212, 252, 144))
    PLAYER_DEATH_ROW_COLORS = ((121, 181, 236), (87, 139, 201), (47, 90, 160), (87, 139, 201), (121, 181, 236))
    EXPLOSION_ROW_COLORS = (
        (150, 50, 137), (50, 50, 176), (112, 0, 20), (236, 236, 236),
        (236, 236, 236), (236, 236, 236), (236, 236, 236), (171, 135, 50),
    )
    BUILDING_COLOR = (150, 113, 26)
    BUILDING_TOP_COLOR = (162, 128, 238)
    SCORE_COLOR = (87, 139, 201)
    HUD_FLASH_COLOR = (82, 82, 82)
    # Greyscale equivalents ALE shows during color loss (colors not listed here are already grey)
    COLOR_LOSS_GREYS = {
        (112, 0, 92): 44, (150, 113, 26): 114, (87, 139, 201): 131, (162, 128, 238): 151,
        (121, 181, 236): 169, (72, 72, 194): 86, (47, 90, 160): 85, (201, 154, 92): 161,
        (160, 107, 50): 116, (183, 92, 176): 129, (68, 116, 182): 109, (236, 194, 128): 199,
        (128, 235, 180): 197, (212, 252, 144): 228, (72, 176, 110): 137, (140, 172, 72): 151,
        (147, 111, 223): 135, (150, 50, 137): 90, (50, 50, 176): 64, (112, 0, 20): 36,
        (171, 135, 50): 136,
    }
    # ALE draws the 75 point enemy 2px to the right of its position
    ENEMY_RENDER_X_OFFSETS = (0, 0, 2, 0)

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
        asset_config = [
            {'name': 'background', 'type': 'background', 'data': padded_background},
            {
                'name': 'player',
                'type': 'group',
                'data': [self._rows_to_rgba(_decode_bitmap(rows, 14, row_repeat=2), self.PLAYER_ROW_COLORS)
                         for rows in self.PLAYER_FRAMES],
            },
            {
                'name': 'player_death',
                'type': 'group',
                'data': [self._rows_to_rgba(_decode_bitmap(rows, 16, row_repeat=2), self.PLAYER_DEATH_ROW_COLORS)
                         for rows in self.PLAYER_DEATH_FRAMES],
            },
            {'name': 'building', 'type': 'group', 'data': self._create_building_sprites()},
            {'name': 'enemy', 'type': 'group', 'data': self._load_enemy_sprites(sprite_path)},
            {
                'name': 'enemy_explosion',
                'type': 'group',
                'data': [self._rows_to_rgba(_decode_bitmap(rows, 16, row_repeat=2), self.EXPLOSION_ROW_COLORS)
                         for rows in self.ENEMY_EXPLOSION_FRAMES],
            },
            {'name': 'missile', 'type': 'single', 'file': 'missile.npy'},
            {'name': 'life', 'type': 'single', 'file': 'life.npy'},
            {
                'name': 'score_digits',
                'type': 'digits',
                'data': jnp.stack([self._rows_to_rgba(_decode_bitmap(rows, 6), (self.SCORE_COLOR,) * 8)
                                   for rows in self.SCORE_DIGITS]),
            },
            {'name': 'hud_flash', 'type': 'procedural', 'data': self._create_hud_flash_sprite()},
        ]

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Same palette ids, but every color replaced by the grey ALE shows during color loss
        grey_palette = np.array(self.PALETTE)
        for i, color in enumerate(grey_palette):
            grey = self.COLOR_LOSS_GREYS.get(tuple(int(c) for c in color[:3]))
            if grey is not None:
                grey_palette[i, :3] = grey
        self.GREY_PALETTE = jnp.array(grey_palette, dtype=self.PALETTE.dtype)

        self.score_digit_spacing = 8

    @staticmethod
    def _rows_to_rgba(bitmap: np.ndarray, row_colors) -> jnp.ndarray:
        """Colors a bitmap with one color per row (rows are repeated to match the bitmap height)."""
        colors = np.repeat(np.array(row_colors, dtype=np.uint8), bitmap.shape[0] // len(row_colors), axis=0)
        rgba = np.zeros(bitmap.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = colors[:, None, :]
        rgba[..., 3] = np.where(bitmap, 255, 0)
        rgba[~bitmap] = 0
        return jnp.array(rgba)

    def _load_enemy_sprites(self, sprite_path: str) -> list:
        """Loads the color enemy sprites (25, 50, 75, 100 points) and adds the second 25 point animation frame."""
        sprites = []
        for points in (25, 50, 75, 100):
            sprite = np.load(os.path.join(sprite_path, f"enemy_{points}_color.npy"))
            # The sprite files contain the sky color around the enemy, make it transparent and crop
            opaque = np.logical_and(sprite[..., 3] > 0, np.any(sprite[..., :3] != self.SKY_COLOR, axis=-1))
            sprite = np.where(opaque[..., None], sprite, 0).astype(np.uint8)
            rows = np.nonzero(opaque.any(axis=1))[0]
            cols = np.nonzero(opaque.any(axis=0))[0]
            sprites.append(sprite[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])
        # The 25 point enemy waves its antenna: the top bar alternates between two positions
        flapped = sprites[0].copy()
        flapped[:2] = np.roll(flapped[:2], 4, axis=1)
        sprites.append(flapped)
        return [jnp.array(sprite) for sprite in sprites]

    def _create_building_sprites(self) -> list:
        sprites = []
        for stage, cells in enumerate(_BUILDING_CELLS):
            bitmap = np.repeat(np.repeat(cells, 2, axis=0), 4, axis=1)
            row_colors = [self.BUILDING_COLOR] * bitmap.shape[0]
            if stage == 0:
                # Differently colored roof tops on the undamaged building
                row_colors[2:4] = [self.BUILDING_TOP_COLOR] * 2
            rgba = self._rows_to_rgba(bitmap, row_colors)
            sprites.append(rgba)
        return sprites

    def _create_hud_flash_sprite(self) -> jnp.ndarray:
        hud_height = AirRaidConstants.HEIGHT - AirRaidConstants.HUD_Y
        hud_flash = np.zeros((hud_height, AirRaidConstants.WIDTH, 4), dtype=np.uint8)
        hud_flash[...] = (*self.HUD_FLASH_COLOR, 255)
        return jnp.array(hud_flash)

    def _load_and_pad_background(self, sprite_path: str) -> jnp.ndarray:
        background = self.jr.loadFrame(os.path.join(sprite_path, "background.npy"))
        # The background file holds the greyscale sky, the game's actual sky is purple
        is_sky = jnp.all(background == jnp.array([44, 44, 44, 255], dtype=background.dtype), axis=-1)
        background = jnp.where(is_sky[..., None], jnp.array([*self.SKY_COLOR, 255], dtype=background.dtype), background)
        # The playfield starts 4 rows below the top of the screen, everything below it is black
        top_rows = AirRaidConstants.HUD_Y - background.shape[0]
        top_padding = jnp.zeros((top_rows, background.shape[1], 4), dtype=background.dtype)
        top_padding = top_padding.at[:, :, 3].set(255)
        background = jnp.concatenate([top_padding, background], axis=0)
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

        building_masks = self.SHAPE_MASKS["building"]
        enemy_masks = self.SHAPE_MASKS["enemy"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        player_masks = self.SHAPE_MASKS["player"]
        player_death_masks = self.SHAPE_MASKS["player_death"]
        missile_mask = self.SHAPE_MASKS["missile"]
        life_mask = self.SHAPE_MASKS["life"]
        score_digit_masks = self.SHAPE_MASKS["score_digits"]

        def render_building(i, raster_in):
            building_mask = building_masks[state.building_damage[i]]
            building_x = state.building_x[i]
            # Buildings wrap around the screen edges
            raster_out = self.jr.render_at_clipped(raster_in, building_x, AirRaidConstants.BUILDING_INITIAL_Y, building_mask)
            return self.jr.render_at_clipped(
                raster_out, building_x - AirRaidConstants.WIDTH, AirRaidConstants.BUILDING_INITIAL_Y, building_mask
            )

        raster = jax.lax.fori_loop(0, AirRaidConstants.NUM_BUILDINGS, render_building, raster)

        enemy_x_offsets = jnp.array(self.ENEMY_RENDER_X_OFFSETS, dtype=jnp.int32)

        def render_enemy(i, raster_in):
            is_active = state.enemy_active[i] == 1
            enemy_type = jnp.clip(state.enemy_type[i], 0, 3)
            # The 25 point enemy alternates between two animation frames every 2 frames
            sprite_index = jnp.where(
                jnp.logical_and(enemy_type == 0, (state.step_counter // 2) % 2 == 1), 4, enemy_type
            )
            enemy_mask = enemy_masks[sprite_index]
            enemy_x = state.enemy_x[i] + enemy_x_offsets[enemy_type]
            render_result = self.jr.render_at_clipped(raster_in, enemy_x, state.enemy_y[i], enemy_mask)

            # Recently shot enemies show a short explosion animation
            explosion_age = AirRaidConstants.ENEMY_KILL_RESPAWN_DELAY - state.enemy_timer[i]
            is_exploding = jnp.logical_and(
                jnp.logical_not(is_active),
                explosion_age < AirRaidConstants.ENEMY_EXPLOSION_FRAMES,
            )
            explosion_frame = jnp.clip(explosion_age // 3, 0, 2)
            explosion_result = self.jr.render_at_clipped(
                raster_in, state.enemy_x[i], state.enemy_y[i], explosion_masks[explosion_frame]
            )
            return jnp.where(is_active, render_result, jnp.where(is_exploding, explosion_result, raster_in))

        enemy_raster = jax.lax.fori_loop(0, AirRaidConstants.TOTAL_ENEMIES, render_enemy, raster)
        # Enemies sink behind the horizon above the player's flight level
        rows = jnp.arange(raster.shape[0])[:, None]
        raster = jnp.where(rows < AirRaidConstants.ENEMY_HORIZON_Y, enemy_raster, raster)

        # The ship's rotor animation alternates every 8 frames
        player_frame = (state.step_counter // 8) % 2
        player_rendered = self.jr.render_at_clipped(raster, state.player_x, state.player_y, player_masks[player_frame])
        raster = jnp.where(state.player_visible == 1, player_rendered, raster)

        # A destroyed ship is shown briefly hit and then exploding
        death_timer = state.player_death_timer
        death_frame = jnp.where(
            death_timer <= AirRaidConstants.PLAYER_HIT_FRAMES,
            0,
            1 + ((death_timer - AirRaidConstants.PLAYER_HIT_FRAMES - 1) // 4) % 2,
        )
        death_rendered = self.jr.render_at_clipped(
            raster, state.player_x, state.player_y + 1, player_death_masks[death_frame]
        )
        show_death = jnp.logical_and(
            jnp.logical_and(state.player_visible == 0, death_timer > 0),
            death_timer <= AirRaidConstants.PLAYER_DEATH_FRAMES,
        )
        raster = jnp.where(show_death, death_rendered, raster)

        def render_player_missile(i, raster_in):
            render_result = self.jr.render_at_clipped(raster_in, state.player_missile_x[i], state.player_missile_y[i], missile_mask)
            return jnp.where(state.player_missile_active[i] == 1, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.NUM_PLAYER_MISSILES, render_player_missile, raster)

        def render_enemy_missile(i, raster_in):
            render_result = self.jr.render_at_clipped(raster_in, state.enemy_missile_x[i], state.enemy_missile_y[i], missile_mask)
            return jnp.where(state.enemy_missile_active[i] == 1, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.NUM_ENEMY_MISSILES, render_enemy_missile, raster)

        # The area below the city flashes when a building is hit
        flash_rendered = self.jr.render_at(raster, 0, AirRaidConstants.HUD_Y, self.SHAPE_MASKS["hud_flash"])
        raster = jnp.where(state.flash_counter > 0, flash_rendered, raster)

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
            AirRaidConstants.SCORE_X,
            AirRaidConstants.SCORE_Y,
            score_digits,
            score_digit_masks,
            start_index,
            num_to_render,
            spacing=self.score_digit_spacing,
            max_digits_to_render=6,
            right_align=True,
        )

        # Reserve ships are shown below the city
        lives = state.player_lives

        def render_life(i, raster_in):
            icon_x = AirRaidConstants.LIFE_X + i * AirRaidConstants.LIFE_SPACING
            render_result = self.jr.render_at(raster_in, icon_x, AirRaidConstants.LIFE_Y, life_mask)
            return jnp.where(i < lives - 1, render_result, raster_in)

        raster = jax.lax.fori_loop(0, AirRaidConstants.MAX_PLAYER_LIVES - 1, render_life, raster)

        # ALE shows the whole frame in greyscale while the enemy bomb slot is in use
        palette = jnp.where(state.color_loss == 1, self.GREY_PALETTE, self.PALETTE)
        return self.jr.render_from_palette(raster, palette)
