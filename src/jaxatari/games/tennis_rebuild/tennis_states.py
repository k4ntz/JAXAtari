from typing import NamedTuple
import jax.numpy as jnp
import chex

# TODO: due to the movement properties of the ball in Tennis, the velocity values should probably represent how frequent ticks are skipped..
# Define state container
class TennisState(NamedTuple):
    player_x: chex.Array  # Player x position
    player_y: chex.Array  # Player y position
    player_direction: chex.Array  # Player direction (0 for right, 1 for left)
    enemy_x: chex.Array  # Enemy x position
    enemy_y: chex.Array  # Enemy y position
    enemy_direction: chex.Array  # Enemy direction (0 for right, 1 for left)
    ball_x: chex.Array  # Ball x position
    ball_y: chex.Array  # Ball y position
    ball_z: chex.Array  # Ball height/shadow
    ball_curve_counter: chex.Array
    ball_x_dir: chex.Array  # Ball x direction
    ball_y_dir: chex.Array  # Ball y direction
    ball_movement_tick: chex.Array  # Ball movement tick
    ball_curve: chex.Array
    ball_start: chex.Array
    ball_end: chex.Array
    shadow_x: chex.Array  # Shadow x position
    shadow_y: chex.Array  # Shadow y position
    player_round_score: chex.Array
    enemy_round_score: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    round_overtime: chex.Array  # boolean array that is only true if the round is in overtime
    game_overtime: chex.Array # boolean array that is only true if the game is in overtime
    serving: chex.Array  # boolean for serve state
    side_switch_counter: chex.Array # tracks side switch cycle position
    just_hit: chex.Array # boolean for just hit state
    player_side: chex.Array  # 0 if player on top side; 1 if player on bottom side
    ball_was_infield: chex.Array
    current_tick: chex.Array
    ball_y_tick: chex.Array
    ball_x_pattern_idx: chex.Array  # Array holding the x-movement pattern
    ball_x_counter: chex.Array  # Current index in the pattern
    player_hit: chex.Array
    enemy_hit: chex.Array


# Observation and Info containers
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class TennisObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    ball: EntityPosition
    ball_shadow: EntityPosition
    player_round_score: jnp.ndarray
    enemy_round_score: jnp.ndarray
    player_score: jnp.ndarray
    enemy_score: jnp.ndarray

class TennisInfo(NamedTuple):
    serving: jnp.ndarray
    player_side: jnp.ndarray
    current_tick: jnp.ndarray
    ball_direction: jnp.ndarray