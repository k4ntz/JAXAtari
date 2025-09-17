import os
from functools import partial
from typing import NamedTuple, Tuple, List, Dict

import jax
import jax.numpy as jnp
import chex
import pygame
from dataclasses import dataclass

import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.rendering.jax_rendering_utils as aj
from jaxatari.renderers import JAXGameRenderer


def load_sprite_frame(path: str) -> chex.Array:
    import numpy as np
    if os.path.exists(path):
        return jnp.array(np.load(path))
    return jnp.array([])  # Return empty array instead of None


@dataclass
class GameConfig:
    """All static configuration parameters for the game."""
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    SKY_COLOR: Tuple[int, int, int] = (100, 149, 237)
    WATER_COLOR: Tuple[int, int, int] = (60, 60, 160)
    WATER_Y_START: int = 64
    RESET: int = 18

    # Player and Rod/Hook
    P1_START_X: int = 20
    P2_START_X: int = 124
    PLAYER_Y: int = 34
    ROD_Y: int = 44  # Y position where rod extends horizontally

    # Rod mechanics
    MIN_ROD_LENGTH: int = 10  # Minimum rod extension
    START_ROD_LENGTH: int = 20  # Starting rod length for natural default position
    MAX_ROD_LENGTH: int = 50  # Maximum rod extension
    ROD_SPEED: float = 1.5  # Speed of rod extension/retraction

    # Fish death line - how far below the rod the fish must be brought to score
    FISH_DEATH_LINE_OFFSET: int = 15  # Increase this to lower the death line

    HOOK_WIDTH: int = 3
    HOOK_HEIGHT: int = 5
    HOOK_SPEED_V: float = 1.0
    REEL_SLOW_SPEED: float = 0.5
    REEL_FAST_SPEED: float = 1.5
    LINE_Y_START: int = 48
    LINE_Y_END: int = 160
    AUTO_LOWER_SPEED: float = 2.0
    # Physics
    Acceleration: float = 0.2
    Damping: float = 0.85

    # Boundaries
    LEFT_BOUNDARY: float = 10
    RIGHT_BOUNDARY: float = 115

    # Fish
    FISH_WIDTH: int = 8
    FISH_HEIGHT: int = 7
    FISH_SPEED: float = 0.8
    NUM_FISH: int = 6
    FISH_ROW_YS: Tuple[int] = (85, 101, 117, 133, 149, 165)
    FISH_ROW_SCORES: Tuple[int] = (2, 2, 4, 4, 6, 6)

    # Shark
    SHARK_WIDTH: int = 16
    SHARK_HEIGHT: int = 7
    SHARK_SPEED: float = 1.0
    SHARK_Y: int = 68
    SHARK_BURST_SPEED: float = 2
    SHARK_BURST_DURATION: int = 240 # Frames
    SHARK_BURST_CHANCE: float = 0.005 # percentage

class PlayerState(NamedTuple):
    rod_length: chex.Array  # Length of horizontal rod extension
    hook_y: chex.Array  # Vertical position of hook (relative to rod end)
    score: chex.Array
    hook_state: chex.Array  # 0=free, 1=hooked/reeling slow, 2=reeling fast, 3=auto-lowering
    hooked_fish_idx: chex.Array
    hook_velocity_y: chex.Array  # Vertical velocity
    hook_x_offset: chex.Array  # Horizontal offset from rod end due to water resistance


class GameState(NamedTuple):
    p1: PlayerState
    p2: PlayerState
    fish_positions: chex.Array
    fish_directions: chex.Array
    fish_active: chex.Array
    shark_x: chex.Array
    shark_dir: chex.Array
    shark_burst_timer: chex.Array
    reeling_priority: chex.Array
    time: chex.Array
    game_over: chex.Array
    key: jax.random.PRNGKey


class FishingDerbyObservation(NamedTuple):
    player1_hook_xy: chex.Array
    fish_xy: chex.Array
    shark_x: chex.Array
    score: chex.Array


class FishingDerbyInfo(NamedTuple):
    p1_score: int
    p2_score: int
    time: int
    all_rewards: chex.Array


# Game Logic
class FishingDerby(JaxEnvironment):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
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
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]

    def _get_hook_position(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        """Calculate the actual hook position based on rod length and hook depth."""
        cfg = self.config
        rod_end_x = player_x + player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(10)) -> Tuple[FishingDerbyObservation, GameState]:
        key, fish_key = jax.random.split(key)

        p1_state = PlayerState(
            rod_length=jnp.array(float(self.config.START_ROD_LENGTH)),
            # Start hook in the water: at water surface relative to rod
            hook_y=jnp.array(float(self.config.WATER_Y_START - self.config.ROD_Y)),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0),
            hook_x_offset=jnp.array(0.0)
        )

        p2_state = PlayerState(
            rod_length=jnp.array(float(self.config.START_ROD_LENGTH)),
            hook_y=jnp.array(float(self.config.WATER_Y_START - self.config.ROD_Y)),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0),
            hook_x_offset=jnp.array(0.0)
        )

        fish_x = jax.random.uniform(fish_key, (self.config.NUM_FISH,), minval=10.0,
                                    maxval=self.config.SCREEN_WIDTH - 20.0)
        fish_y = jnp.array(self.config.FISH_ROW_YS, dtype=jnp.float32)

        state = GameState(
            p1=p1_state, p2=p2_state,
            fish_positions=jnp.stack([fish_x, fish_y], axis=1),
            fish_directions=jax.random.choice(key, jnp.array([-1.0, 1.0]), (self.config.NUM_FISH,)),
            fish_active=jnp.ones(self.config.NUM_FISH, dtype=jnp.bool_),
            shark_x=jnp.array(self.config.SCREEN_WIDTH / 2.0),
            shark_dir=jnp.array(1.0),
            shark_burst_timer=jnp.array(0),
            reeling_priority=jnp.array(-1),
            time=jnp.array(0),
            game_over=jnp.array(False),
            key=key
        )
        return self._get_observation(state), state

    def _get_observation(self, state: GameState) -> FishingDerbyObservation:
        hook_x, hook_y = self._get_hook_position(self.config.P1_START_X, state.p1)
        return FishingDerbyObservation(
            player1_hook_xy=jnp.array([hook_x, hook_y]),
            fish_xy=state.fish_positions,
            shark_x=state.shark_x,
            score=state.p1.score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, old_state: GameState, new_state: GameState) -> chex.Array:
        """Return scalar reward for player 1 (the main player)."""
        p1_delta = new_state.p1.score - old_state.p1.score
        return p1_delta

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, old_state: GameState, new_state: GameState) -> chex.Array:
        """Return all rewards as an array (for multi-reward scenarios)."""
        p1_delta = new_state.p1.score - old_state.p1.score
        p2_delta = new_state.p2.score - old_state.p2.score
        return jnp.array([p1_delta, p2_delta])

    def _get_done(self, state: GameState) -> bool:
        return state.game_over

    def _get_info(self, state: GameState) -> FishingDerbyInfo:
        return FishingDerbyInfo(
            p1_score=state.p1.score,
            p2_score=state.p2.score,
            time=state.time,
            all_rewards=self._get_all_rewards(state, state)  # Use _get_all_rewards for info
        )


    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> Tuple[
        FishingDerbyObservation, GameState, chex.Array, bool, FishingDerbyInfo]:
        """Processes one frame of the game and returns the full tuple."""
        new_state = self._step_logic(state, action)
        observation = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return observation, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> chex.Array:
        """Render the current game state."""
        renderer = FishingDerbyRenderer()
        return renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space of the environment."""
        return spaces.Dict({
            "player1_hook_xy": spaces.Box(
                low=jnp.array([0.0, 0.0], dtype=jnp.float32),
                high=jnp.array([self.config.SCREEN_WIDTH, self.config.SCREEN_HEIGHT], dtype=jnp.float32),
                shape=(2,),
                dtype=jnp.float32
            ),
            "fish_xy": spaces.Box(
                low=jnp.array([[0.0, 0.0]] * self.config.NUM_FISH, dtype=jnp.float32),
                high=jnp.array([[self.config.SCREEN_WIDTH, self.config.SCREEN_HEIGHT]] * self.config.NUM_FISH,
                               dtype=jnp.float32),
                shape=(self.config.NUM_FISH, 2),
                dtype=jnp.float32
            ),
            "shark_x": spaces.Box(
                low=jnp.array(0.0, dtype=jnp.float32),
                high=jnp.array(self.config.SCREEN_WIDTH, dtype=jnp.float32),
                shape=(),
                dtype=jnp.float32
            ),
            "score": spaces.Box(
                low=jnp.array(0.0, dtype=jnp.float32),
                high=jnp.array(99.0, dtype=jnp.float32),
                shape=(),
                dtype=jnp.float32
            )
        })

    def image_space(self) -> spaces.Space:
        """Returns the image space of the environment."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.SCREEN_HEIGHT, self.config.SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

    def obs_to_flat_array(self, obs: FishingDerbyObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""
        return jnp.concatenate([
            obs.player1_hook_xy,  # 2 values: hook x, y
            obs.fish_xy.flatten(),  # 12 values: 6 fish * 2 coordinates each
            jnp.array([obs.shark_x, obs.score])  # 2 values: shark x, score
        ])

    def _step_logic(self, state: GameState, p1_action: int) -> GameState:
        """The core logic for a single game step, returning only the new state."""
        cfg = self.config

        def reset_branch(_):
            _, new_state = self.reset(state.key)
            return new_state

        def safe_set_at(arr, idx, value, pred):
            def do_set(a):
                return a.at[idx].set(value)

            return jax.lax.cond(pred, do_set, lambda a: a, arr)

        def game_branch(_):
            # Fish movement with random direction changes
            key = state.key
            key, fish_key = jax.random.split(key)
            change_direction_prob = 0.01  # 1% chance per frame to change direction

            # Check for random direction changes
            should_change_dir = jax.random.uniform(fish_key, (cfg.NUM_FISH,)) < change_direction_prob

            # Fish movement
            new_fish_x = state.fish_positions[:, 0] + state.fish_directions * cfg.FISH_SPEED

            # Direction changes: either from hitting boundaries OR random changes
            hit_boundary = (new_fish_x < 0) | (new_fish_x > cfg.SCREEN_WIDTH - cfg.FISH_WIDTH)
            change_dir = hit_boundary | should_change_dir

            new_fish_dirs = jnp.where(change_dir, -state.fish_directions, state.fish_directions)
            new_fish_x = jnp.clip(new_fish_x, 0, cfg.SCREEN_WIDTH - cfg.FISH_WIDTH)
            new_fish_pos = state.fish_positions.at[:, 0].set(new_fish_x)

            # Shark movement
            key, shark_key = jax.random.split(key)

            # Check for random speed burst initiation
            should_start_burst = (state.shark_burst_timer == 0) & (
                    jax.random.uniform(shark_key) < cfg.SHARK_BURST_CHANCE)
            new_burst_timer = jnp.where(should_start_burst, cfg.SHARK_BURST_DURATION, state.shark_burst_timer)

            # Determine current shark speed
            is_bursting = new_burst_timer > 0
            current_shark_speed = jnp.where(is_bursting, cfg.SHARK_BURST_SPEED, cfg.SHARK_SPEED)

            # Random direction changes
            key, shark_dir_key = jax.random.split(key)
            change_direction_prob = 0.005  # Chance to turn direction randomly
            should_change_dir = jax.random.uniform(shark_dir_key) < change_direction_prob

            # Calculate where shark would move
            potential_shark_x = state.shark_x + state.shark_dir * current_shark_speed

            # Check boundaries and random direction changes
            would_hit_left = potential_shark_x <= cfg.LEFT_BOUNDARY
            would_hit_right = potential_shark_x >= cfg.RIGHT_BOUNDARY
            would_hit_boundary = would_hit_left | would_hit_right

            # Change direction for either boundary hit OR random change
            should_change_direction = would_hit_boundary | should_change_dir
            new_shark_dir = jnp.where(should_change_direction, -state.shark_dir, state.shark_dir)

            # Move with new direction if direction changed, otherwise use original movement
            new_shark_x = jnp.where(
                should_change_direction,
                jnp.clip(state.shark_x + new_shark_dir * current_shark_speed, cfg.LEFT_BOUNDARY, cfg.RIGHT_BOUNDARY),
                jnp.clip(potential_shark_x, cfg.LEFT_BOUNDARY, cfg.RIGHT_BOUNDARY)
            )

            # Update burst timer (decrement if active)
            new_burst_timer = jnp.where(new_burst_timer > 0, new_burst_timer - 1, 0)

            # Player 1 Rod and Hook Logic
            p1 = state.p1

            # Track previous rod length to detect horizontal movement
            prev_rod_length = p1.rod_length

            # Define hook position limits that will be used throughout
            min_hook_y = 0.0  # At rod level
            max_hook_y = cfg.LINE_Y_END - cfg.ROD_Y  # Maximum depth
            water_surface_hook_y = float(cfg.WATER_Y_START - cfg.ROD_Y)  # Water surface level

            # Rod length control (horizontal extension)
            rod_change = 0.0
            rod_change = jnp.where(p1_action == Action.RIGHT, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.LEFT, -cfg.ROD_SPEED, rod_change)

            new_rod_length = jnp.clip(
                p1.rod_length + rod_change,
                cfg.MIN_ROD_LENGTH,
                cfg.MAX_ROD_LENGTH
            )

            # Calculate horizontal rod movement for water resistance
            rod_velocity_x = new_rod_length - prev_rod_length

            # Water resistance physics for hook horizontal offset
            # When hook is in water (hook_y > 0), apply water resistance
            in_water = p1.hook_y > 0.0
            # Stronger resistance produces more noticeable trailing/bend
            water_resistance_factor = jnp.where(in_water, 1.6, 0.0)

            # Hook follows rod movement with delay due to water resistance
            # The deeper the hook, the more resistance
            depth_factor = jnp.clip(p1.hook_y / (cfg.LINE_Y_END - cfg.ROD_Y), 0.0, 1.0)
            resistance_multiplier = 1.0 + depth_factor * 4.0

            # Calculate target horizontal offset based on rod movement
            target_offset = -rod_velocity_x * water_resistance_factor * resistance_multiplier

            # Hook offset follows target with damping (creates lag effect)
            offset_damping = 0.75  # less damping -> larger, quicker offsets
            new_hook_x_offset = p1.hook_x_offset * offset_damping + target_offset * (1.0 - offset_damping)

            # When not in water, offset decays back to 0 faster
            new_hook_x_offset = jnp.where(in_water, new_hook_x_offset, p1.hook_x_offset * 0.6)

            # Hook vertical movement and auto-lowering logic
            def auto_lower_hook(_):
                # Move hook down towards water surface
                new_y = p1.hook_y + cfg.AUTO_LOWER_SPEED
                new_vel_y = 0.0  # Override velocity during auto-lowering

                # Check if hook reached water surface
                hook_reached_water = new_y >= water_surface_hook_y
                final_state = jnp.where(hook_reached_water, 0, p1.hook_state)  # Return to free state
                final_y = jnp.where(hook_reached_water, water_surface_hook_y, new_y)

                return final_y, new_vel_y, final_state

            def normal_hook_movement(_):
                # Normal hook movement (only when free - hook_state == 0)
                can_move_vertically = (p1.hook_state == 0)
                change = jnp.where(can_move_vertically & (p1_action == Action.DOWN), +cfg.Acceleration, 0.0)
                change = jnp.where(can_move_vertically & (p1_action == Action.UP), -cfg.Acceleration, change)

                # Update hook velocity with damping
                new_vel_y = p1.hook_velocity_y * cfg.Damping + change

                # Calculate hook position limits
                min_y = 0.0  # At rod level
                max_y = cfg.LINE_Y_END - cfg.ROD_Y  # Maximum depth

                # Update hook position
                new_y = jnp.clip(
                    p1.hook_y + new_vel_y,
                    min_y,
                    max_y
                )

                # Kill velocity if hitting bounds
                final_vel_y = jnp.where(
                    (new_y == min_y) | (new_y == max_y),
                    0.0,
                    new_vel_y
                )

                return new_y, final_vel_y, p1.hook_state

            # Choose between auto-lowering and normal movement
            new_hook_y, new_hook_velocity_y, p1_hook_state = jax.lax.cond(
                p1.hook_state == 3,
                auto_lower_hook,
                normal_hook_movement,
                operand=None
            )

            # Get actual hook position in world coordinates
            hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, PlayerState(
                rod_length=new_rod_length,
                hook_y=new_hook_y,
                score=p1.score,
                hook_state=p1_hook_state,
                hooked_fish_idx=p1.hooked_fish_idx,
                hook_velocity_y=new_hook_velocity_y,
                hook_x_offset=new_hook_x_offset
            ))

            # Collision and Game Logic
            fish_active, reeling_priority = state.fish_active, state.reeling_priority
            can_hook = (p1_hook_state == 0)  # Use updated hook state
            hook_collides_fish = (jnp.abs(new_fish_pos[:, 0] - hook_x) < cfg.FISH_WIDTH) & (
                    jnp.abs(new_fish_pos[:, 1] - hook_y) < cfg.FISH_HEIGHT)
            valid_hook_targets = can_hook & fish_active & hook_collides_fish

            hooked_fish_idx, did_hook_fish = jnp.argmax(valid_hook_targets), jnp.any(valid_hook_targets)

            p1_hook_state = jnp.where(did_hook_fish, 1, p1_hook_state)
            p1_hooked_fish_idx = jnp.where(did_hook_fish, hooked_fish_idx, p1.hooked_fish_idx)
            fish_active = fish_active.at[hooked_fish_idx].set(
                jnp.where(did_hook_fish, False, fish_active[hooked_fish_idx])
            )
            reeling_priority = jnp.where(did_hook_fish & (reeling_priority == -1), 0, reeling_priority)

            # Fast reel with FIRE button
            can_reel_fast = (p1_action == Action.FIRE) & (p1_hook_state == 1) & (
                    (reeling_priority == -1) | (reeling_priority == 0))
            p1_hook_state = jnp.where(can_reel_fast, 2, p1_hook_state)
            reeling_priority = jnp.where(can_reel_fast, 0, reeling_priority)

            # Reeling mechanics (moves hook upward toward rod)
            reel_speed = jnp.where(p1_hook_state == 2, cfg.REEL_FAST_SPEED, cfg.REEL_SLOW_SPEED)
            new_hook_y = jnp.where(p1_hook_state > 0,
                                   jnp.clip(new_hook_y - reel_speed, min_hook_y, max_hook_y),
                                   new_hook_y)
            new_hook_velocity_y = jnp.where(p1_hook_state > 0, 0.0, new_hook_velocity_y)

            # Update hook position after reeling
            hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, PlayerState(
                rod_length=new_rod_length,
                hook_y=new_hook_y,
                score=p1.score,
                hook_state=p1_hook_state,
                hooked_fish_idx=p1_hooked_fish_idx,
                hook_velocity_y=new_hook_velocity_y,
                hook_x_offset=new_hook_x_offset
            ))

            # Hooked fish follows the hook
            hooked_fish_pos = jnp.array([hook_x, hook_y])
            has_hook = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0)
            new_fish_pos = safe_set_at(new_fish_pos, p1_hooked_fish_idx, hooked_fish_pos, has_hook)

            # Scoring and collision detection
            p1_score, key = p1.score, state.key
            shark_collides = (p1_hook_state > 0) & (jnp.abs(hook_x - new_shark_x) < cfg.SHARK_WIDTH) & (
                    jnp.abs(hook_y - cfg.SHARK_Y) < cfg.SHARK_HEIGHT)
            scored_fish = (p1_hook_state > 0) & (hook_y <= cfg.ROD_Y + 5)  # Fish reaches near the rod
            reset_hook = shark_collides | scored_fish

            # Handle scoring and update fish state
            prev_idx = p1_hooked_fish_idx
            fish_scores = jnp.array(cfg.FISH_ROW_SCORES)
            p1_score += jnp.where(scored_fish, fish_scores[p1_hooked_fish_idx], 0)

            # Fish respawn logic - simpler version
            def respawn_fish(all_pos, all_dirs, idx, key):
                kx, kdir = jax.random.split(key)
                new_x = jax.random.uniform(kx, minval=10.0, maxval=cfg.SCREEN_WIDTH - cfg.FISH_WIDTH)
                new_y = jnp.array(cfg.FISH_ROW_YS, dtype=jnp.float32)[idx]
                new_pos = all_pos.at[idx].set(jnp.array([new_x, new_y]))
                new_dir = all_dirs.at[idx].set(jax.random.choice(kdir, jnp.array([-1.0, 1.0])))
                return new_pos, new_dir

            key, respawn_key = jax.random.split(key)
            do_respawn = reset_hook & (prev_idx >= 0)
            new_fish_pos, new_fish_dirs = jax.lax.cond(
                do_respawn,
                lambda _: respawn_fish(new_fish_pos, new_fish_dirs, prev_idx, respawn_key),
                lambda _: (new_fish_pos, new_fish_dirs),
                operand=None
            )

            # Reset hook state and fish activity when hook is reset
            p1_hook_state = jnp.where(reset_hook, 3, p1_hook_state)  # Set to auto-lowering state
            p1_hooked_fish_idx = jnp.where(reset_hook, -1, p1_hooked_fish_idx)  # Clear hooked fish
            new_hook_x_offset = jnp.where(reset_hook, 0.0, new_hook_x_offset)

            # CRITICAL FIX: Reactivate the fish when it's scored/eaten by shark
            fish_active = jnp.where(
                reset_hook & (prev_idx >= 0),
                fish_active.at[prev_idx].set(True),  # Reactivate the fish
                fish_active
            )

            game_over = (p1_score >= 99) | (state.p2.score >= 99)

            return GameState(
                p1=PlayerState(
                    rod_length=new_rod_length,
                    hook_y=new_hook_y,
                    score=p1_score,
                    hook_state=p1_hook_state,
                    hooked_fish_idx=p1_hooked_fish_idx,
                    hook_velocity_y=new_hook_velocity_y,
                    hook_x_offset=new_hook_x_offset
                ),
                p2=state.p2,
                fish_positions=new_fish_pos,
                fish_directions=new_fish_dirs,
                fish_active=fish_active,
                shark_x=new_shark_x,
                shark_dir=new_shark_dir,
                shark_burst_timer=new_burst_timer,
                reeling_priority=reeling_priority,
                time=state.time + 1,
                game_over=game_over,
                key=key
            )

        return jax.lax.cond(p1_action == cfg.RESET, reset_branch, game_branch, state)


def normalize_frame(frame: chex.Array, target_shape: Tuple[int, int, int]) -> chex.Array:
    """Crop or pad a sprite to the target shape with transparent (255) background."""
    h, w, c = frame.shape
    th, tw, tc = target_shape
    assert c == tc

    # Crop if larger
    frame = frame[:min(h, th), :min(w, tw), :]

    # Pad if smaller (with 255 = white = transparent)
    pad_h = th - frame.shape[0]
    pad_w = tw - frame.shape[1]

    frame = jnp.pad(
        frame,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=255
    )
    return frame


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    sprite_paths = {
        'background': os.path.join(MODULE_DIR, "sprites/fishingderby/background.npy"),
        'player1': os.path.join(MODULE_DIR, "sprites/fishingderby/player1.npy"),
        'player2': os.path.join(MODULE_DIR, "sprites/fishingderby/player2.npy"),
        'shark1': os.path.join(MODULE_DIR, "sprites/fishingderby/shark_new_1.npy"),
        'shark2': os.path.join(MODULE_DIR, "sprites/fishingderby/shark_new_2.npy"),
        'fish1': os.path.join(MODULE_DIR, "sprites/fishingderby/fish1.npy"),
        'fish2': os.path.join(MODULE_DIR, "sprites/fishingderby/fish2.npy"),
        'sky': os.path.join(MODULE_DIR, "sprites/fishingderby/sky.npy"),
        **{f"score_{i}": os.path.join(MODULE_DIR, f"sprites/fishingderby/score_{i}.npy") for i in range(10)}
    }

    sprites = {}
    for name, path in sprite_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sprite file not found: {path}")
        sprite = aj.loadFrame(path, transpose=False)
        if sprite is None:
            raise ValueError(f"Failed to load sprite: {path}")
        sprites[name] = sprite

    # Normalize shark frames
    shark_sprites = [sprites['shark1'], sprites['shark2']]
    max_shape = (
        max(s.shape[0] for s in shark_sprites),
        max(s.shape[1] for s in shark_sprites),
        shark_sprites[0].shape[2]
    )
    sprites['shark1'] = normalize_frame(sprites['shark1'], max_shape)
    sprites['shark2'] = normalize_frame(sprites['shark2'], max_shape)

    # Normalize fish frames
    fish_sprites = [sprites['fish1'], sprites['fish2']]
    max_shape = (
        max(s.shape[0] for s in fish_sprites),
        max(s.shape[1] for s in fish_sprites),
        fish_sprites[0].shape[2]
    )
    sprites['fish1'] = normalize_frame(sprites['fish1'], max_shape)
    sprites['fish2'] = normalize_frame(sprites['fish2'], max_shape)

    # Score digits
    score_digits = jnp.stack([sprites[f'score_{i}'] for i in range(10)])

    return (
        sprites['background'], sprites['player1'], sprites['player2'],
        sprites['shark1'], sprites['shark2'], sprites['fish1'], sprites['fish2'],
        sprites['sky'], score_digits
    )


class FishingDerbyRenderer(JAXGameRenderer):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        (
            self.SPRITE_BG, self.SPRITE_PLAYER1, self.SPRITE_PLAYER2,
            self.SPRITE_SHARK1, self.SPRITE_SHARK2, self.SPRITE_FISH1,
            self.SPRITE_FISH2, self.SPRITE_SKY, self.SPRITE_SCORE_DIGITS
        ) = load_sprites()

    def _get_hook_position(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        """Calculate the actual hook position based on rod length and hook depth."""
        cfg = self.config
        rod_end_x = player_x + player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> chex.Array:
        cfg = self.config
        raster = jnp.zeros((cfg.SCREEN_HEIGHT, cfg.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Draw sky and water
        raster = raster.at[:cfg.WATER_Y_START, :, :].set(jnp.array(cfg.SKY_COLOR, dtype=jnp.uint8))
        raster = raster.at[cfg.WATER_Y_START:, :, :].set(jnp.array(cfg.WATER_COLOR, dtype=jnp.uint8))

        # Draw players
        raster = self._render_at(raster, cfg.P1_START_X, cfg.PLAYER_Y, self.SPRITE_PLAYER1)
        raster = self._render_at(raster, cfg.P2_START_X, cfg.PLAYER_Y, self.SPRITE_PLAYER2)

        # Draw fishing line with water resistance physics
        p1_rod_end_x = cfg.P1_START_X + state.p1.rod_length

        # Get actual hook position including water resistance offset
        hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, state.p1)

        # Draw horizontal part of the rod
        raster = self._render_line(raster, cfg.P1_START_X + 10, cfg.ROD_Y, p1_rod_end_x, cfg.ROD_Y,
                                   color=(139, 69, 19))  # Brown rod

        # Water resistance sag - only applies when hook is in water and below surface
        in_water = state.p1.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)

        # Only apply horizontal movement sag when there's recent horizontal movement AND in water
        has_horizontal_movement = jnp.abs(state.p1.hook_x_offset) > 0.1
        offset_sag = jnp.where(in_water & has_horizontal_movement,
                               jnp.abs(state.p1.hook_x_offset) * 1.0,
                               0.0)

        # Vertical movement sag (normal gravity sag) - only when in water
        max_hook_y = cfg.LINE_Y_END - cfg.ROD_Y
        water_depth_ratio = jnp.clip((state.p1.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / max_hook_y, 0.0, 1.0)

        # Apply gravity sag only to underwater portion
        gravity_sag_base = 6.0 * jnp.exp(-2.0 * water_depth_ratio)
        gravity_sag = jnp.where(in_water & has_horizontal_movement,
                                gravity_sag_base,
                                jnp.where(in_water, gravity_sag_base * 0.2, 0.0))

        # Combine both types of sag - only when in water
        total_sag = jnp.where(in_water, gravity_sag + offset_sag, 0.0)

        # Reduce sag when reeling (line is under tension)
        is_reeling = state.p1.hook_state > 0
        tension_multiplier = jnp.where(is_reeling, 0.2, 1.0)
        final_sag = total_sag * tension_multiplier

        # Define start and end points for the line
        line_start = jnp.array([p1_rod_end_x, cfg.ROD_Y])
        line_end = jnp.array([hook_x, hook_y])

        # Draw straight line if above water, saggy line if in water
        raster = jax.lax.cond(
            in_water,
            lambda r: self._render_saggy_line(r, line_start, line_end, final_sag, color=(200, 200, 200),
                                              num_segments=12),
            lambda r: self._render_line(r, line_start[0], line_start[1], line_end[0], line_end[1],
                                        color=(200, 200, 200)),
            raster
        )

        # Player 2 rod and line (simplified for now)
        raster = self._render_line(raster, cfg.P2_START_X + 2, cfg.PLAYER_Y + 10, state.p2.rod_length + cfg.P2_START_X,
                                   cfg.ROD_Y + state.p2.hook_y)

        # Draw shark
        shark_frame = jax.lax.cond((state.time // 4) % 2 == 0, lambda: self.SPRITE_SHARK1, lambda: self.SPRITE_SHARK2)
        raster = self._render_at(raster, state.shark_x, cfg.SHARK_Y, shark_frame, flip_h=state.shark_dir < 0)

        # Draw fish
        fish_frame = jax.lax.cond((state.time // 5) % 2 == 0, lambda: self.SPRITE_FISH1, lambda: self.SPRITE_FISH2)

        def draw_one_fish(i, r):
            pos, direction, active = state.fish_positions[i], state.fish_directions[i], state.fish_active[i]
            return jax.lax.cond(active,
                                lambda r_in: self._render_at(r_in, pos[0], pos[1], fish_frame, flip_h=direction > 0),
                                lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, cfg.NUM_FISH, draw_one_fish, raster)

        # Draw hooked fish - FIXED: Only draw when fish is actually hooked and active
        def draw_hooked_p1(r):
            hook_x_pos, hook_y_pos = self._get_hook_position(cfg.P1_START_X, state.p1)
            return self._render_at(r, hook_x_pos, hook_y_pos, fish_frame)

        should_draw_hooked = (state.p1.hook_state > 0) & (state.p1.hooked_fish_idx >= 0) & (state.p1.hook_state != 3)
        raster = jax.lax.cond(should_draw_hooked, draw_hooked_p1, lambda r: r, raster)

        raster = self._render_score(raster, state.p1.score, 50, 20)
        raster = self._render_score(raster, state.p2.score, 100, 20)

        return raster

    def _render_score(self, raster, score, x, y):
        s1, s0 = score // 10, score % 10
        digit1_sprite, digit0_sprite = self.SPRITE_SCORE_DIGITS[s1], self.SPRITE_SCORE_DIGITS[s0]
        raster = self._render_at(raster, x, y, digit1_sprite)
        raster = self._render_at(raster, x + 7, y, digit0_sprite)
        return raster

    @staticmethod
    @jax.jit
    def _render_at(raster, x, y, sprite, flip_h=False):
        sprite_rgb = sprite[:, :, :3]
        h, w = sprite.shape[0], sprite.shape[1]
        x, y = jnp.round(x).astype(jnp.int32), jnp.round(y).astype(jnp.int32)
        sprite_to_draw = jnp.where(flip_h, jnp.fliplr(sprite_rgb), sprite_rgb)

        has_alpha = sprite.shape[2] > 3
        if has_alpha:
            alpha = sprite[:, :, 3:4]
            alpha = jnp.where(flip_h, jnp.fliplr(alpha), alpha)
            mask = alpha > 0
        else:
            is_black = jnp.all(sprite_to_draw == 0, axis=-1, keepdims=True)
            is_white = jnp.all(sprite_to_draw == 255, axis=-1, keepdims=True)
            mask = ~(is_black | is_white)

        region = jax.lax.dynamic_slice(raster, (y, x, 0), (h, w, 3))
        patch = jnp.where(mask, sprite_to_draw, region)

        return jax.lax.dynamic_update_slice(raster, patch, (y, x, 0))

    @staticmethod
    @jax.jit
    def _render_line(raster, x0, y0, x1, y1, color=(200, 200, 200)):
        x0, y0, x1, y1 = jnp.round(jnp.array([x0, y0, x1, y1])).astype(jnp.int32)
        dx, sx, dy, sy = jnp.abs(x1 - x0), jnp.sign(x1 - x0), -jnp.abs(y1 - y0), jnp.sign(y1 - y0)
        err = dx + dy
        color_uint8 = jnp.array(color, dtype=jnp.uint8)

        def loop_body(carry):
            x, y, r, e = carry
            safe_y, safe_x = jnp.clip(y, 0, r.shape[0] - 1), jnp.clip(x, 0, r.shape[1] - 1)
            r = r.at[safe_y, safe_x, :].set(color_uint8)
            e2 = 2 * e
            e_new = jnp.where(e2 >= dy, e + dy, e)
            x_new = jnp.where(e2 >= dy, x + sx, x)
            e_final = jnp.where(e2 <= dx, e_new + dx, e_new)
            y_new = jnp.where(e2 <= dx, y + sy, y)
            return x_new, y_new, r, e_final

        def loop_cond(carry):
            return ~((carry[0] == x1) & (carry[1] == y1))

        _, _, raster, _ = jax.lax.while_loop(loop_cond, loop_body, (x0, y0, raster, err))
        # Ensure the last pixel is drawn
        safe_y1, safe_x1 = jnp.clip(y1, 0, raster.shape[0] - 1), jnp.clip(x1, 0, raster.shape[1] - 1)
        raster = raster.at[safe_y1, safe_x1, :].set(color_uint8)
        return raster

    @staticmethod
    @partial(jax.jit, static_argnums=(5,))
    def _render_saggy_line(raster, p_start, p_end, sag_amount, color, num_segments=10):
        """Renders a fishing line with realistic catenary-like sag."""

        # Create 't' values from 0.0 to 1.0 to parameterize the curve
        t = jnp.linspace(0.0, 1.0, num_segments + 1)

        # Linearly interpolate between start and end points
        points = jax.vmap(lambda i: p_start + i * (p_end - p_start))(t)

        # Calculate line properties
        line_vector = p_end - p_start
        line_length = jnp.linalg.norm(line_vector) + 1e-8  # Add small epsilon to avoid division by zero

        # More realistic sag calculation combining parabolic and catenary effects
        # Parabolic component (primary gravity effect)
        parabolic_sag = 4.0 * t * (1.0 - t)  # Maximum at t=0.5, zero at endpoints

        # Catenary-like component for more realistic physics
        # Uses hyperbolic cosine shape but simplified for performance
        catenary_factor = jnp.cosh(3.0 * (t - 0.5)) - 1.0
        normalized_catenary = catenary_factor / (jnp.max(catenary_factor) + 1e-8)

        # Combine both components with line length influence
        # Shorter lines sag more relative to their length
        length_factor = jnp.clip(line_length / 50.0, 0.5, 2.0)  # Scale based on line length
        total_sag = sag_amount * (0.8 * parabolic_sag + 0.2 * normalized_catenary) / length_factor

        # Use JAX-compatible conditional for sag direction
        # For nearly vertical lines (fishing lines), apply horizontal sag
        is_nearly_vertical = jnp.abs(line_vector[0]) < 0.1

        # Calculate perpendicular direction for non-vertical lines
        line_direction_norm = line_vector / line_length
        perpendicular = jnp.array([-line_direction_norm[1], line_direction_norm[0]])

        # Use jnp.where for JAX-compatible conditional logic
        sag_offsets_x = jnp.where(is_nearly_vertical, total_sag, total_sag * perpendicular[0])
        sag_offsets_y = jnp.where(is_nearly_vertical, jnp.zeros_like(total_sag), total_sag * perpendicular[1])

        # Apply sag offsets to points
        points = points.at[:, 0].add(sag_offsets_x)
        points = points.at[:, 1].add(sag_offsets_y)

        # Draw segments
        def draw_segment(i, current_raster):
            p1 = points[i]
            p2 = points[i + 1]
            return FishingDerbyRenderer._render_line(
                current_raster, p1[0], p1[1], p2[0], p2[1], color
            )

        raster = jax.lax.fori_loop(0, num_segments, draw_segment, raster)
        return raster


def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    reset = keys[pygame.K_r]

    if reset:
        return jnp.array(GameConfig.RESET)

    # Map keyboard inputs to Atari actions
    is_up = up and not down
    is_down = down and not up
    is_left = left and not right
    is_right = right and not left

    if fire:
        if is_up and is_left: return jnp.array(Action.UPLEFTFIRE)
        if is_up and is_right: return jnp.array(Action.UPRIGHTFIRE)
        if is_down and is_left: return jnp.array(Action.DOWNLEFTFIRE)
        if is_down and is_right: return jnp.array(Action.DOWNRIGHTFIRE)
        if is_up: return jnp.array(Action.UPFIRE)
        if is_down: return jnp.array(Action.DOWNFIRE)
        if is_left: return jnp.array(Action.LEFTFIRE)
        if is_right: return jnp.array(Action.RIGHTFIRE)
        return jnp.array(Action.FIRE)
    else:
        if is_up and is_left: return jnp.array(Action.UPLEFT)
        if is_up and is_right: return jnp.array(Action.UPRIGHT)
        if is_down and is_left: return jnp.array(Action.DOWNLEFT)
        if is_down and is_right: return jnp.array(Action.DOWNRIGHT)
        if is_up: return jnp.array(Action.UP)
        if is_down: return jnp.array(Action.DOWN)
        if is_left: return jnp.array(Action.LEFT)
        if is_right: return jnp.array(Action.RIGHT)

    return jnp.array(Action.NOOP)


if __name__ == "__main__":
    pygame.init()
    game = FishingDerby()
    renderer = FishingDerbyRenderer()
    jitted_step = jax.jit(game.step)
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    (_, curr_state) = game.reset(reset_key)

    scaling = 4
    screen = pygame.display.set_mode((GameConfig.SCREEN_WIDTH * scaling, GameConfig.SCREEN_HEIGHT * scaling))
    pygame.display.set_caption("JAX Fishing Derby")

    running = True
    frame_by_frame = False

    clock = pygame.time.Clock()

    print("Controls:")
    print("WASD or Arrow Keys - Move hook/rod")
    print("SPACE - Fast reel (when fish is hooked)")
    print("R - Reset game")
    print("F - Toggle frame-by-frame mode")
    print("N - Next frame (in frame-by-frame mode)")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    print(f"Frame-by-frame mode: {'ON' if frame_by_frame else 'OFF'}")
                elif event.key == pygame.K_n and frame_by_frame:
                    action = get_human_action()
                    key, step_key = jax.random.split(curr_state.key)
                    (_, curr_state, _, _, _) = jitted_step(curr_state, action)

        if not frame_by_frame:
            action = get_human_action()
            key, step_key = jax.random.split(curr_state.key)
            (_, curr_state, _, _, _) = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)
        aj.update_pygame(screen, raster, scaling, GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)

        # Display game info
        if curr_state.time % 60 == 0:  # Update every second
            print(
                f"Player 1 Score: {curr_state.p1.score}, Player 2 Score: {curr_state.p2.score}, Time: {curr_state.time}")

        clock.tick(60)

    pygame.quit()

    # run with: python scripts/play.py --game fishingderby --record my_record_file.npz