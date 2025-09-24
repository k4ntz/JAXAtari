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
    SKY_COLOR: Tuple[int, int, int] = (117, 128, 240)
    WATER_COLOR: Tuple[int, int, int] = (24, 26, 167)
    WATER_Y_START: int = 60
    WATER_SHIMMER_HEIGHT: int = 16
    RESET: int = 18

    # Player and Rod/Hook
    P1_START_X: int = 9
    P2_START_X: int = 135
    PLAYER_Y: int = 23
    ROD_Y: int = 38  # Y position where rod extends horizontally
    FISH_SCORING_Y: int = 78

    # Rod mechanics
    MIN_ROD_LENGTH_X: int = 23  # Minimum horizontal rod extension
    START_ROD_LENGTH_X: int = 23  # Starting horizontal rod length
    MAX_ROD_LENGTH_X: int = 65  # Maximum horizontal extension
    P2_MIN_ROD_LENGTH_X: int = 7  # Reduce this to allow less leftward extension
    P2_MAX_ROD_LENGTH_X: int = 46  # Increase this to allow more rightward extension
    MIN_HOOK_DEPTH_Y: int = 0  # Minimum vertical hook depth
    START_HOOK_DEPTH_Y: int = 40  # Starting vertical hook depth
    MAX_HOOK_DEPTH_Y: int = 160  # Maximum vertical extension to reach bottom fish

    ROD_SPEED: float = 1
    # Fish death line - how far below the rod the fish must be brought to score
    FISH_DEATH_LINE_OFFSET: int = 20  # Increase this to lower the death line

    HOOK_WIDTH: int = 3
    HOOK_HEIGHT: int = 5
    HOOK_SPEED_V: float = 10
    REEL_SLOW_SPEED: float = 1
    REEL_FAST_SPEED: float = 2
    LINE_Y_START: int = 48
    LINE_Y_END: int = 180
    AUTO_LOWER_SPEED: float = 2.0
    # Physics
    Acceleration: float = 0.2
    Damping: float = 0.85
    SLOW_REEL_PERIOD: int = 6  # slow reel: 1 px every 4 frames
    MAX_HOOKED_WOBBLE_DX: float = 0.9  # max extra sideways dx per frame when hooked
    WOBBLE_FREQ_BASE: float = 0.10  # base wobble frequency
    WOBBLE_FREQ_RANGE: float = 0.06  # extra freq added with depth
    WOBBLE_AMP_BASE: float = 0.05  # base wobble dx (px/frame)
    WOBBLE_AMP_RANGE: float = 0.20  # extra wobble dx with depth

    # Occasional downward tugs by row (px/frame) when you’re NOT reeling on this frame
    FISH_PULL_PER_ROW: Tuple[float, ...] = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20)

    # Boundaries
    LEFT_BOUNDARY: float = 10
    RIGHT_BOUNDARY: float = 115
    # Fish
    FISH_WIDTH: int = 8
    FISH_HEIGHT: int = 7
    FISH_SPEED: float = 0.4
    NUM_FISH: int = 6
    FISH_ROW_YS: Tuple[int] = (95, 111, 127, 143, 159, 175)
    FISH_ROW_SCORES: Tuple[int] = (2, 2, 4, 4, 6, 6)
    # When hooked
    HOOKED_FISH_SPEED_MULTIPLIER: float = 1.5
    HOOKED_FISH_TURN_PROBABILITY: float = 0.04
    HOOKED_FISH_BOUNDARY_ENABLED: bool = True
    HOOKED_FISH_BOUNDARY_PADDING: int = 20 # Max distance from line

    # Normal swimming
    FISH_BASE_TURN_PROBABILITY: float = 0.01  # 1% chance to change direction

    # Turning cooldown for hooked fish
    HOOKED_FISH_TURNING_COOLDOWN: int = 30  # frames before hooked fish can turn again


    # Shark
    SHARK_WIDTH: int = 16
    SHARK_HEIGHT: int = 7
    SHARK_SPEED: float = 0.3
    SHARK_Y: int = 78
    SHARK_BURST_SPEED: float = 1.5
    SHARK_BURST_DURATION: int = 300 # Frames
    SHARK_BURST_CHANCE: float = 0.001 # percentage

class PlayerState(NamedTuple):
    rod_length: chex.Array  # Length of horizontal rod extension
    hook_y: chex.Array  # Vertical position of hook (relative to rod end)
    score: chex.Array
    hook_state: chex.Array  # 0=free, 1=hooked/reeling slow, 2=reeling fast, 3=auto-lowering
    hooked_fish_idx: chex.Array
    hook_velocity_y: chex.Array  # Vertical velocity
    hook_x_offset: chex.Array  # Horizontal offset from rod end due to water resistance
    display_score: chex.Array  # animated display score
    score_animation_timer: chex.Array  # control animation timing
    line_segments_x: chex.Array  # X positions of line segments for trailing effect


class GameState(NamedTuple):
    p1: PlayerState
    p2: PlayerState
    fish_positions: chex.Array
    fish_directions: chex.Array
    fish_active: chex.Array
    fish_turn_cooldowns: chex.Array  # Per-fish turning cooldown timers
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

    def _get_hook_position_p2(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        """Calculate the actual hook position for Player 2 based on rod length and hook depth."""
        cfg = self.config
        # Player 2's rod extends leftward, so subtract rod length from starting position
        rod_end_x = player_x - player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(10)) -> Tuple[FishingDerbyObservation, GameState]:
        key, fish_key = jax.random.split(key)

        p1_state = PlayerState(
            rod_length=jnp.array(float(self.config.START_ROD_LENGTH_X)),
            hook_y=jnp.array(float(self.config.START_HOOK_DEPTH_Y)),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0),
            hook_x_offset=jnp.array(0.0),
            display_score=jnp.array(0),
            score_animation_timer=jnp.array(0),
            line_segments_x=jnp.zeros(8)  # Initialize 8 line segments for trailing effect
        )

        p2_state = PlayerState(
            rod_length=jnp.array(float(self.config.START_ROD_LENGTH_X)),
            hook_y=jnp.array(float(self.config.WATER_Y_START - self.config.ROD_Y)),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0),
            hook_x_offset=jnp.array(0.0),
            display_score=jnp.array(0),
            score_animation_timer=jnp.array(0),
            line_segments_x=jnp.zeros(8)  # Initialize 8 line segments for trailing effect
        )

        fish_x = jax.random.uniform(fish_key, (self.config.NUM_FISH,), minval=self.config.LEFT_BOUNDARY,
                                    maxval=self.config.RIGHT_BOUNDARY)
        fish_y = jnp.array(self.config.FISH_ROW_YS, dtype=jnp.float32)

        state = GameState(
            p1=p1_state, p2=p2_state,
            fish_positions=jnp.stack([fish_x, fish_y], axis=1),
            fish_directions=jax.random.choice(key, jnp.array([-1.0, 1.0]), (self.config.NUM_FISH,)),
            fish_active=jnp.ones(self.config.NUM_FISH, dtype=jnp.bool_),
            fish_turn_cooldowns=jnp.zeros(self.config.NUM_FISH, dtype=jnp.int32),  # Initialize cooldowns
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
        p2_hook_x, p2_hook_y = self._get_hook_position_p2(self.config.P2_START_X, state.p2)
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
    def step(self, state: GameState, action: int, p2_action: int = -1) -> Tuple[
        FishingDerbyObservation, GameState, chex.Array, bool, FishingDerbyInfo]:
        """Processes one frame of the game and returns the full tuple."""

        key, p2_key = jax.random.split(state.key)

        def strategic_p2_ai():
            """
            Improved P2 AI that mimics 1980s-style game AI behavior:
            - Only targets bottom 3 fish rows (worth 4-6 points each)
            - Fast reels when shark is on the left side
            - Properly retracts rod when reeling in fish
            - Efficient patterns that would work within 6502 constraints
            """
            cfg = self.config
            p2_state = state.p2
            fish_pos = state.fish_positions
            fish_active = state.fish_active

            # Get current hook position
            p2_hook_x, p2_hook_y = self._get_hook_position_p2(cfg.P2_START_X, p2_state)

            # Current rod and hook state
            rod_length = p2_state.rod_length
            hook_state = p2_state.hook_state
            hooked_fish = p2_state.hooked_fish_idx

            # Simple state machine based on hook state

            # STATE 1: Fish is hooked - reel it in
            def reeling_behavior():
                """When fish is hooked, retract rod and fast reel when shark is on left"""
                # Fast reel when shark is on the left side of the screen
                # This gives P2 advantage since P2 is on the right
                shark_on_left = state.shark_x < (cfg.SCREEN_WIDTH / 2)

                # Also fast reel when very close to scoring
                close_to_scoring = p2_hook_y < cfg.FISH_SCORING_Y + 10

                # Use fast reel (FIRE button) when shark is on left or near scoring
                use_fast_reel = shark_on_left | close_to_scoring

                # For P2: RIGHT = retract rod (decrease length, pulls toward player)
                # Combine with UP for reeling and FIRE for fast reel
                # Note: UPRIGHT would extend rod (wrong!), we need UP then RIGHT separately
                # But since we can only send one action, prioritize reeling up with occasional retract

                # Alternate between pure UP and RIGHTFIRE/RIGHT to achieve both reeling and retracting
                should_retract = (state.time % 3) == 0  # Retract every 3rd frame

                return jnp.where(
                    use_fast_reel,
                    jnp.where(should_retract, Action.RIGHTFIRE, Action.UPFIRE),  # Fast reel + retract
                    jnp.where(should_retract, Action.RIGHT, Action.UP)  # Normal reel + retract
                )

            # STATE 2: Hook is free - find nearest catchable fish
            def fishing_behavior():
                """Targets bottom 3 fish, but switches to top 3 if all bottom fish are far left"""

                # Calculate distances to all active fish
                # Use Manhattan distance for simplicity (6502-friendly)
                fish_distances = jnp.abs(fish_pos[:, 0] - p2_hook_x) + jnp.abs(fish_pos[:, 1] - p2_hook_y)

                # Mask out inactive fish (set their distance to infinity)
                fish_distances = jnp.where(fish_active, fish_distances, jnp.inf)

                # Check if P2's hook is deep (at bottom area) and all bottom 3 fish are on left side
                hook_at_bottom = p2_hook_y > 140  # Hook is in bottom third of water

                # Check positions of bottom 3 fish (indices 3, 4, 5)
                bottom_fish_x = fish_pos[3:6, 0]  # X positions of bottom 3 fish
                all_bottom_fish_left = jnp.all(bottom_fish_x < 60)  # All on left half

                # If at bottom and all bottom fish are far left, target top 3 instead
                should_target_top = hook_at_bottom & all_bottom_fish_left & fish_active[3] & fish_active[4] & \
                                    fish_active[5]

                # Set infinite distance for fish we're not targeting
                fish_distances = jnp.where(
                    should_target_top,
                    # Target top 3 (0,1,2), ignore bottom 3 (3,4,5)
                    fish_distances.at[3].set(jnp.inf).at[4].set(jnp.inf).at[5].set(jnp.inf),
                    # Normal: target bottom 3 (3,4,5), ignore top 3 (0,1,2)
                    fish_distances.at[0].set(jnp.inf).at[1].set(jnp.inf).at[2].set(jnp.inf)
                )

                # Find best target
                nearest_fish_idx = jnp.argmin(fish_distances)
                nearest_fish_x = fish_pos[nearest_fish_idx, 0]
                nearest_fish_y = fish_pos[nearest_fish_idx, 1]

                # Simple targeting thresholds
                CLOSE_ENOUGH_X = 8.0  # Horizontal proximity to attempt catch
                CLOSE_ENOUGH_Y = 8.0  # Vertical proximity to attempt catch

                # Calculate position deltas
                dx = nearest_fish_x - p2_hook_x
                dy = nearest_fish_y - p2_hook_y

                # Determine if we're close enough to try catching
                close_x = jnp.abs(dx) < CLOSE_ENOUGH_X
                close_y = jnp.abs(dy) < CLOSE_ENOUGH_Y
                in_catch_zone = close_x & close_y

                # Rod control logic (P2's rod extends leftward)
                # Need to extend rod (increase length) to move hook left
                # Need to retract rod (decrease length) to move hook right
                need_left = dx < -CLOSE_ENOUGH_X  # Fish is to the left
                need_right = dx > CLOSE_ENOUGH_X  # Fish is to the right
                need_down = dy > CLOSE_ENOUGH_Y
                need_up = dy < -CLOSE_ENOUGH_Y

                # Simple priority system: horizontal first, then vertical
                # This mimics simple 1980s AI decision trees

                # If we're in the catch zone, just go down slowly to hook
                action_in_zone = jnp.where(
                    dy > 2,  # If slightly below, go down
                    Action.DOWN,
                    Action.NOOP  # Otherwise wait for fish
                )

                # Horizontal + Vertical combinations (8-directional movement)
                action_diagonal = jnp.where(
                    need_left & need_down,
                    Action.DOWNLEFT,  # Down + extend rod (left for P2)
                    jnp.where(
                        need_left & need_up,
                        Action.UPLEFT,  # Up + extend rod
                        jnp.where(
                            need_right & need_down,
                            Action.DOWNRIGHT,  # Down + retract rod (right for P2)
                            jnp.where(
                                need_right & need_up,
                                Action.UPRIGHT,  # Up + retract rod
                                Action.NOOP
                            )
                        )
                    )
                )

                # Pure horizontal movement
                action_horizontal = jnp.where(
                    need_left,
                    Action.LEFT,  # Extend rod left
                    jnp.where(
                        need_right,
                        Action.RIGHT,  # Retract rod right
                        Action.NOOP
                    )
                )

                # Pure vertical movement
                action_vertical = jnp.where(
                    need_down,
                    Action.DOWN,
                    jnp.where(
                        need_up,
                        Action.UP,
                        Action.NOOP
                    )
                )

                # Decision priority (simpler for 1980s hardware):
                # 1. If in catch zone, try to hook
                # 2. If need diagonal movement, use it
                # 3. If need horizontal only, do that
                # 4. If need vertical only, do that
                # 5. Otherwise, NOOP

                return jnp.where(
                    in_catch_zone,
                    action_in_zone,
                    jnp.where(
                        (need_left | need_right) & (need_up | need_down),
                        action_diagonal,
                        jnp.where(
                            need_left | need_right,
                            action_horizontal,
                            action_vertical
                        )
                    )
                )

            # STATE 3: Auto-lowering after catch - wait
            def auto_lower_behavior():
                """During auto-lower, position rod for next catch"""
                # Simple behavior: return rod to neutral position
                rod_is_neutral = jnp.abs(rod_length - cfg.START_ROD_LENGTH_X) < 5

                return jnp.where(
                    rod_length > cfg.START_ROD_LENGTH_X + 5,
                    Action.RIGHT,  # Retract if extended
                    jnp.where(
                        rod_length < cfg.START_ROD_LENGTH_X - 5,
                        Action.LEFT,  # Extend if retracted
                        Action.NOOP
                    )
                )

            # Main state machine dispatcher
            ai_action = jnp.where(
                hook_state == 3,  # Auto-lowering
                auto_lower_behavior(),
                jnp.where(
                    hooked_fish >= 0,  # Has a fish hooked
                    reeling_behavior(),
                    fishing_behavior()  # Looking for fish
                )
            )

            return ai_action
        # Player 2 is always controlled by the AI.
        p2_action = strategic_p2_ai()
        state = state._replace(key=key)

        new_state = self._step_logic(state, action, p2_action)
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

    def _is_fire_action(self, a: int) -> chex.Array:
        """True for FIRE and any directional FIRE combo (e.g., UPFIRE, RIGHTFIRE...)."""
        return (
                (a == Action.FIRE)
                | (a == Action.UPFIRE)
                | (a == Action.DOWNFIRE)
                | (a == Action.LEFTFIRE)
                | (a == Action.RIGHTFIRE)
                | (a == Action.UPLEFTFIRE)
                | (a == Action.UPRIGHTFIRE)
                | (a == Action.DOWNLEFTFIRE)
                | (a == Action.DOWNRIGHTFIRE)
        )

    def _step_logic(self, state: GameState, p1_action: int, p2_action: int) -> GameState:
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
            # Shorthand
            p1 = state.p1
            p2 = state.p2

            # RNG
            key = state.key
            key, fish_key = jax.random.split(key)

            # ==== Fish movement (unverändert zu vorher) =========================================
            base_change_prob = cfg.FISH_BASE_TURN_PROBABILITY
            p1_hooked_idx = state.p1.hooked_fish_idx
            change_probs = jnp.full(cfg.NUM_FISH, base_change_prob)
            change_probs = jnp.where(
                jnp.arange(cfg.NUM_FISH) == p1_hooked_idx,
                cfg.HOOKED_FISH_TURN_PROBABILITY,
                change_probs
            )
            fish_speeds = jnp.full(cfg.NUM_FISH, cfg.FISH_SPEED)
            fish_speeds = jnp.where(
                jnp.arange(cfg.NUM_FISH) == p1_hooked_idx,
                cfg.FISH_SPEED * cfg.HOOKED_FISH_SPEED_MULTIPLIER,
                fish_speeds
            )
            new_fish_x = state.fish_positions[:, 0] + state.fish_directions * fish_speeds
            # Adjust boundary check to allow fish to move further right
            effective_right_boundary = cfg.RIGHT_BOUNDARY + 24
            hit_boundary = (new_fish_x <= cfg.LEFT_BOUNDARY) | (new_fish_x >= effective_right_boundary)
            should_change_dir_random = jax.random.uniform(fish_key, (cfg.NUM_FISH,)) < change_probs
            can_turn_due_to_cooldown = state.fish_turn_cooldowns <= 0
            should_change_dir = (can_turn_due_to_cooldown & should_change_dir_random) | hit_boundary
            new_cooldowns = jnp.where(
                should_change_dir,
                cfg.HOOKED_FISH_TURNING_COOLDOWN,
                jnp.maximum(0, state.fish_turn_cooldowns - 1)
            )
            new_fish_dirs = jnp.where(should_change_dir, -state.fish_directions, state.fish_directions)
            new_fish_x = jnp.clip(new_fish_x, cfg.LEFT_BOUNDARY, effective_right_boundary)
            new_fish_pos = state.fish_positions.at[:, 0].set(new_fish_x)

            # ==== Shark movement (unverändert zu vorher) ========================================
            key, shark_key = jax.random.split(key)
            should_start_burst = (state.shark_burst_timer == 0) & (
                        jax.random.uniform(shark_key) < cfg.SHARK_BURST_CHANCE)
            new_burst_timer = jnp.where(should_start_burst, cfg.SHARK_BURST_DURATION, state.shark_burst_timer)
            is_bursting = new_burst_timer > 0
            current_shark_speed = jnp.where(is_bursting, cfg.SHARK_BURST_SPEED, cfg.SHARK_SPEED)
            key, shark_dir_key = jax.random.split(key)
            change_direction_prob = 0.005
            should_change_dir = jax.random.uniform(shark_dir_key) < change_direction_prob
            potential_shark_x = state.shark_x + state.shark_dir * current_shark_speed
            would_hit_left = potential_shark_x <= cfg.LEFT_BOUNDARY
            would_hit_right = potential_shark_x >= cfg.RIGHT_BOUNDARY
            would_hit_boundary = would_hit_left | would_hit_right
            should_change_direction = would_hit_boundary | should_change_dir
            new_shark_dir = jnp.where(should_change_direction, -state.shark_dir, state.shark_dir)
            new_shark_x = jnp.where(
                should_change_direction,
                jnp.clip(state.shark_x + new_shark_dir * current_shark_speed, cfg.LEFT_BOUNDARY, cfg.RIGHT_BOUNDARY),
                jnp.clip(potential_shark_x, cfg.LEFT_BOUNDARY, cfg.RIGHT_BOUNDARY)
            )
            new_burst_timer = jnp.where(new_burst_timer > 0, new_burst_timer - 1, 0)

            # ======== Gemeinsame Konstanten ======================================================

            min_hook_y = 0.0
            max_hook_y = cfg.LINE_Y_END - cfg.ROD_Y
            water_surface_hook_y = float(cfg.WATER_Y_START - cfg.ROD_Y)

            scoring_hook_y = float(cfg.FISH_SCORING_Y - cfg.ROD_Y)

            # ======== P1: Rod horizontal =========================================================
            rod_change = 0.0
            rod_change = jnp.where(p1_action == Action.RIGHT, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.LEFT, -cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.UPRIGHT, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.DOWNRIGHT, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.UPLEFT, -cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.DOWNLEFT, -cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.UPRIGHTFIRE, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.DOWNRIGHTFIRE, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.UPLEFTFIRE, -cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.DOWNLEFTFIRE, -cfg.ROD_SPEED, rod_change)
            new_rod_length = jnp.clip(p1.rod_length + rod_change, cfg.MIN_ROD_LENGTH_X, cfg.MAX_ROD_LENGTH_X)

            # ======== P2: Rod horizontal (spiegelverkehrt) ======================================
            p2_rod_change = 0.0
            p2_rod_change = jnp.where(p2_action == Action.LEFT, +cfg.ROD_SPEED, p2_rod_change)
            p2_rod_change = jnp.where(p2_action == Action.RIGHT, -cfg.ROD_SPEED, p2_rod_change)
            p2_new_rod_length = jnp.clip(p2.rod_length + p2_rod_change, cfg.P2_MIN_ROD_LENGTH_X,cfg.P2_MAX_ROD_LENGTH_X)

            # ======== P1: Wasserwiderstand Hook-X-Offset ========================================
            p1_in_water = p1.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
            rod_end_x_p1 = cfg.P1_START_X + new_rod_length
            target_hook_x_p1 = rod_end_x_p1
            current_hook_x_p1 = cfg.P1_START_X + p1.rod_length + p1.hook_x_offset
            water_resistance_factor = 0.15
            air_recovery_factor = 0.3
            smooth_recovery_factor = 0.08
            actual_rod_change_p1 = new_rod_length - p1.rod_length
            depth_factor_p1 = jnp.clip((p1.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0, 1.0)
            resistance_multiplier_p1 = 1.0 + depth_factor_p1 * 2.0

            def p1_apply_water_resistance():
                resistance = water_resistance_factor * resistance_multiplier_p1
                target_offset = target_hook_x_p1 - rod_end_x_p1
                current_offset = p1.hook_x_offset
                is_moving = jnp.abs(actual_rod_change_p1) > 0.01
                moving_offset = current_offset + (target_offset - current_offset) * resistance
                rod_moving_right = actual_rod_change_p1 > 0
                rod_moving_left = actual_rod_change_p1 < 0
                lag_magnitude = jnp.abs(actual_rod_change_p1) * 0.8 * resistance_multiplier_p1
                directional_lag = jnp.where(rod_moving_right, -lag_magnitude,
                                            jnp.where(rod_moving_left, +lag_magnitude, 0.0))
                moving_result = moving_offset + directional_lag
                stationary_result = current_offset * (1.0 - smooth_recovery_factor)
                return jnp.where(is_moving, moving_result, stationary_result)

            def p1_apply_air_recovery():
                return p1.hook_x_offset * (1.0 - air_recovery_factor)

            new_hook_x_offset = jax.lax.cond(p1_in_water, p1_apply_water_resistance, p1_apply_air_recovery)

            # ======== P2: Wasserwiderstand Hook-X-Offset (spiegeln) =============================
            p2_in_water = p2.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
            rod_end_x_p2 = cfg.P2_START_X - p2_new_rod_length
            target_hook_x_p2 = rod_end_x_p2

            current_hook_x_p2 = cfg.P2_START_X - p2.rod_length + p2.hook_x_offset
            actual_rod_change_p2 = p2_new_rod_length - p2.rod_length
            depth_factor_p2 = jnp.clip((p2.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0, 1.0)
            resistance_multiplier_p2 = 1.0 + depth_factor_p2 * 2.0

            def p2_apply_water_resistance():
                resistance = water_resistance_factor * resistance_multiplier_p2
                target_offset = target_hook_x_p2 - rod_end_x_p2
                current_offset = p2.hook_x_offset
                is_moving = jnp.abs(actual_rod_change_p2) > 0.01
                moving_offset = current_offset + (target_offset - current_offset) * resistance
                # Wichtig: Längen-Änderung >0 bedeutet bei P2, dass der Stab NACH LINKS fährt
                rod_moving_left = actual_rod_change_p2 > 0
                rod_moving_right = actual_rod_change_p2 < 0
                lag_magnitude = jnp.abs(actual_rod_change_p2) * 0.8 * resistance_multiplier_p2
                directional_lag = jnp.where(rod_moving_right, -lag_magnitude,
                                            jnp.where(rod_moving_left, +lag_magnitude, 0.0))
                moving_result = moving_offset + directional_lag
                stationary_result = current_offset * (1.0 - smooth_recovery_factor)
                return jnp.where(is_moving, moving_result, stationary_result)

            def p2_apply_air_recovery():
                return p2.hook_x_offset * (1.0 - air_recovery_factor)

            p2_new_hook_x_offset = jax.lax.cond(p2_in_water, p2_apply_water_resistance, p2_apply_air_recovery)

            # ======== P1: Vertikale Hook-Bewegung + Auto-Lower ==================================
            def p1_auto_lower(_):
                new_y = p1.hook_y + cfg.AUTO_LOWER_SPEED
                new_vel_y = 0.0
                hook_reached_water = new_y >= water_surface_hook_y
                final_state = jnp.where(hook_reached_water, 0, p1.hook_state)
                final_y = jnp.where(hook_reached_water, water_surface_hook_y, new_y)
                return final_y, new_vel_y, final_state

            def p1_normal(_):
                can_move_vertically = (p1.hook_state == 0)
                change = jnp.where(can_move_vertically & (p1_action == Action.DOWN), +cfg.Acceleration, 0.0)
                change = jnp.where(can_move_vertically & (p1_action == Action.UP), -cfg.Acceleration, change)
                change = jnp.where(can_move_vertically & ((p1_action == Action.DOWNLEFT) |
                                                          (p1_action == Action.DOWNRIGHT)), +cfg.Acceleration, change)
                change = jnp.where(can_move_vertically & ((p1_action == Action.UPLEFT) |
                                                          (p1_action == Action.UPRIGHT)), -cfg.Acceleration, change)
                new_vel_y = p1.hook_velocity_y * cfg.Damping + change
                min_y = float(cfg.START_HOOK_DEPTH_Y)
                max_y = float(cfg.MAX_HOOK_DEPTH_Y)
                new_y = jnp.clip(p1.hook_y + new_vel_y, min_y, max_y)
                final_vel_y = jnp.where((new_y == min_y) | (new_y == max_y), 0.0, new_vel_y)
                return new_y, final_vel_y, p1.hook_state

            new_hook_y, new_hook_velocity_y, p1_hook_state = jax.lax.cond(
                p1.hook_state == 3, p1_auto_lower, p1_normal, operand=None
            )

            # P1 Hook-Position (Weltkoordinaten)
            hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, PlayerState(
                rod_length=new_rod_length, hook_y=new_hook_y, score=p1.score, hook_state=p1_hook_state,
                hooked_fish_idx=p1.hooked_fish_idx, hook_velocity_y=new_hook_velocity_y,
                hook_x_offset=new_hook_x_offset,
                display_score=p1.display_score, score_animation_timer=p1.score_animation_timer,
                line_segments_x=p1.line_segments_x
            ))

            # ======== Kollisionslogik / Einhaken P1 =============================================
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
            is_fire_pressed_p1 = self._is_fire_action(p1_action)
            is_reeling_fast_p1 = (p1_hook_state == 1) & is_fire_pressed_p1

            # Fast reel is every frame, slow reel is every cfg.SLOW_REEL_PERIOD frames
            tick_slow = (jnp.bitwise_and(state.time, cfg.SLOW_REEL_PERIOD - 1) == 0)
            reel_tick = jnp.where(is_reeling_fast_p1, True, tick_slow)

            reel_step = jnp.where(p1_hook_state > 0, 1.0, 0.0)  # 1 px per tick
            can_reel = p1_hooked_fish_idx >= 0

            new_hook_y = jnp.where(
                (reel_tick & can_reel),
                jnp.clip(new_hook_y - reel_step, scoring_hook_y, max_hook_y),
                new_hook_y
            )

            # ======== Wenn P1 eine Fisch hat: Fisch folgt / Grenzen / Wobble ====================
            has_hook = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0)

            def p1_update_hooked():
                fish_idx = p1_hooked_fish_idx
                fish_x, fish_y = new_fish_pos[fish_idx, 0], new_fish_pos[fish_idx, 1]
                fish_dir = new_fish_dirs[fish_idx]
                hx, hy = self._get_hook_position(cfg.P1_START_X, PlayerState(
                    rod_length=new_rod_length, hook_y=new_hook_y, score=0, hook_state=0, hooked_fish_idx=-1,
                    hook_velocity_y=0, hook_x_offset=new_hook_x_offset, display_score=0, score_animation_timer=0,
                    line_segments_x=jnp.zeros(8)
                ))
                depth_ratio = jnp.clip((hy - cfg.WATER_Y_START) / (cfg.MAX_HOOK_DEPTH_Y), 0.0, 1.0)
                wobble_freq = cfg.WOBBLE_FREQ_BASE + depth_ratio * cfg.WOBBLE_FREQ_RANGE
                wobble_amp = cfg.WOBBLE_AMP_BASE + depth_ratio * cfg.WOBBLE_AMP_RANGE
                wobble_dx = jnp.sin(state.time * wobble_freq) * wobble_amp
                base_dx = fish_dir * cfg.FISH_SPEED * cfg.HOOKED_FISH_SPEED_MULTIPLIER
                total_dx = jnp.clip(base_dx + wobble_dx, -cfg.MAX_HOOKED_WOBBLE_DX, cfg.MAX_HOOKED_WOBBLE_DX)
                potential_new_x = fish_x + total_dx
                rod_end_x_local = cfg.P1_START_X + new_rod_length
                boundary_min = jnp.maximum(rod_end_x_local - cfg.HOOKED_FISH_BOUNDARY_PADDING, cfg.LEFT_BOUNDARY)
                boundary_max = jnp.minimum(rod_end_x_local + cfg.HOOKED_FISH_BOUNDARY_PADDING, cfg.RIGHT_BOUNDARY)

                def apply_hooked_boundaries():
                    would_hit_left = potential_new_x <= boundary_min
                    would_hit_right = potential_new_x >= boundary_max
                    constrained_x = jnp.clip(potential_new_x, boundary_min, boundary_max)
                    new_direction = jnp.where(would_hit_left | would_hit_right, -fish_dir, fish_dir)
                    elastic_push = jnp.where(would_hit_left, 0.5, jnp.where(would_hit_right, -0.5, 0.0))
                    return constrained_x + elastic_push, new_direction

                def apply_global_boundaries():
                    would_hit_left = potential_new_x <= cfg.LEFT_BOUNDARY
                    would_hit_right = potential_new_x >= cfg.RIGHT_BOUNDARY
                    constrained_x = jnp.clip(potential_new_x, cfg.LEFT_BOUNDARY, cfg.RIGHT_BOUNDARY)
                    new_direction = jnp.where(would_hit_left | would_hit_right, -fish_dir, fish_dir)
                    return constrained_x, new_direction

                new_x, new_fish_direction = jax.lax.cond(
                    cfg.HOOKED_FISH_BOUNDARY_ENABLED, apply_hooked_boundaries, apply_global_boundaries
                )
                updated_pos = new_fish_pos.at[fish_idx, 0].set(new_x)
                updated_pos = updated_pos.at[fish_idx, 1].set(hy - cfg.FISH_HEIGHT / 2.0)
                updated_dirs = new_fish_dirs.at[fish_idx].set(new_fish_direction)

                is_facing_right = new_fish_direction > 0
                mouth_x_offset = jnp.where(is_facing_right, cfg.FISH_WIDTH, 0.0)
                hook_target_x = new_x + mouth_x_offset
                rod_end_x_local = cfg.P1_START_X + new_rod_length
                new_offset = hook_target_x - rod_end_x_local

                row_idx = jnp.clip(fish_idx, 0, len(cfg.FISH_PULL_PER_ROW) - 1).astype(jnp.int32)
                fish_pull = jnp.array(cfg.FISH_PULL_PER_ROW)[row_idx]
                tug_key = jax.random.fold_in(state.key, state.time * 31 + fish_idx)
                do_tug = (jax.random.uniform(tug_key) < (0.08 + 0.04 * depth_ratio)) & (~reel_tick)
                tug_amount = jnp.where(do_tug, fish_pull, 0.0)
                return updated_pos, updated_dirs, new_offset, tug_amount

            tug_amount_p1 = jnp.array(0.0)
            new_fish_pos, new_fish_dirs, new_hook_x_offset, tug_amount_p1 = jax.lax.cond(
                has_hook, lambda _: p1_update_hooked(), lambda _: (new_fish_pos, new_fish_dirs, new_hook_x_offset, 0.0),
                operand=None
            )
            new_hook_y = jnp.clip(new_hook_y + tug_amount_p1, 0.0, max_hook_y)

            # ======== P1 Scoring / Shark-Kollision ==============================================
            p1_score = p1.score
            has_hooked_fish = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0)

            def p1_fpos():
                return new_fish_pos[p1_hooked_fish_idx, 0], new_fish_pos[p1_hooked_fish_idx, 1]

            fish_x_p1, fish_y_p1 = jax.lax.cond(has_hooked_fish, p1_fpos, lambda: (0.0, 0.0))

            # Shark collision logic (unchanged)
            collision_padding = 2.0
            fish_half_w = (cfg.FISH_WIDTH + collision_padding) / 2
            fish_half_h = (cfg.FISH_HEIGHT + collision_padding) / 2
            shark_half_w = (cfg.SHARK_WIDTH + collision_padding) / 2
            shark_half_h = (cfg.SHARK_HEIGHT + collision_padding) / 2
            fish_center_x = fish_x_p1 + cfg.FISH_WIDTH / 2
            fish_center_y = fish_y_p1 + cfg.FISH_HEIGHT / 2
            shark_center_x = new_shark_x + cfg.SHARK_WIDTH / 2
            shark_center_y = cfg.SHARK_Y + cfg.SHARK_HEIGHT / 2
            collides_x = jnp.abs(fish_center_x - shark_center_x) < (fish_half_w + shark_half_w)
            collides_y = jnp.abs(fish_center_y - shark_center_y) < (fish_half_h + shark_half_h)
            shark_collides_p1 = has_hooked_fish & collides_x & collides_y

            scoring_tolerance = 5.0  # pixels above the scoring line where fish still count as scored
            hook_at_surface = hook_y <= (cfg.FISH_SCORING_Y + scoring_tolerance)
            fish_at_surface = fish_y_p1 <= (cfg.FISH_SCORING_Y + scoring_tolerance)

            # Score if either hook OR fish is at/above the scoring line (with tolerance)
            scored_fish_p1 = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0) & (hook_at_surface | fish_at_surface)

            reset_hook_p1 = shark_collides_p1 | scored_fish_p1

            prev_idx_p1 = p1_hooked_fish_idx
            fish_scores = jnp.array(cfg.FISH_ROW_SCORES)
            p1_score += jnp.where(scored_fish_p1, fish_scores[p1_hooked_fish_idx], 0)
            animation_speed = 2
            score_increased_p1 = scored_fish_p1
            new_animation_timer_p1 = jnp.where(score_increased_p1, animation_speed,
                                               jnp.where(p1.score_animation_timer > 0, p1.score_animation_timer - 1, 0))
            should_inc_disp_p1 = (new_animation_timer_p1 == 0) & (p1.display_score < p1_score)
            new_display_score_p1 = jnp.where(should_inc_disp_p1, p1.display_score + 1, p1.display_score)
            new_animation_timer_p1 = jnp.where(should_inc_disp_p1 & (new_display_score_p1 < p1_score),
                                               animation_speed, new_animation_timer_p1)

            def respawn_fish(all_pos, all_dirs, idx, key_local):
                kx, kdir = jax.random.split(key_local)
                new_x = jax.random.uniform(kx, minval=cfg.LEFT_BOUNDARY, maxval=cfg.RIGHT_BOUNDARY)
                new_y = jnp.array(cfg.FISH_ROW_YS, dtype=jnp.float32)[idx]
                new_pos = all_pos.at[idx].set(jnp.array([new_x, new_y]))
                new_dir = all_dirs.at[idx].set(jax.random.choice(kdir, jnp.array([-1.0, 1.0])))
                return new_pos, new_dir

            key, respawn_key_p1 = jax.random.split(key)
            do_respawn_p1 = reset_hook_p1 & (prev_idx_p1 >= 0)
            new_fish_pos, new_fish_dirs = jax.lax.cond(
                do_respawn_p1,
                lambda _: respawn_fish(new_fish_pos, new_fish_dirs, prev_idx_p1, respawn_key_p1),
                lambda _: (new_fish_pos, new_fish_dirs),
                operand=None
            )
            p1_hook_state = jnp.where(reset_hook_p1, 3, p1_hook_state)
            p1_hooked_fish_idx = jnp.where(reset_hook_p1, -1, p1_hooked_fish_idx)
            fish_active = jnp.where(do_respawn_p1, fish_active.at[prev_idx_p1].set(True), fish_active)

            # ======== P2: Vertikale Hook-Bewegung + Auto-Lower ==================================
            def p2_auto_lower(_):
                new_y = p2.hook_y + cfg.AUTO_LOWER_SPEED
                new_vel_y = 0.0
                hook_reached_water = new_y >= water_surface_hook_y
                final_state = jnp.where(hook_reached_water, 0, p2.hook_state)
                final_y = jnp.where(hook_reached_water, water_surface_hook_y, new_y)
                return final_y, new_vel_y, final_state

            def p2_normal(_):
                can_move_vertically = (p2.hook_state == 0)
                change = jnp.where(can_move_vertically & (p2_action == Action.DOWN), +cfg.Acceleration, 0.0)
                change = jnp.where(can_move_vertically & (p2_action == Action.UP), -cfg.Acceleration, change)
                change = jnp.where(can_move_vertically & ((p2_action == Action.DOWNLEFT) |
                                                          (p2_action == Action.DOWNRIGHT)), +cfg.Acceleration, change)
                change = jnp.where(can_move_vertically & ((p2_action == Action.UPLEFT) |
                                                          (p2_action == Action.UPRIGHT)), -cfg.Acceleration, change)
                new_vel_y = p2.hook_velocity_y * cfg.Damping + change
                min_y = float(cfg.START_HOOK_DEPTH_Y)
                max_y = float(cfg.MAX_HOOK_DEPTH_Y)
                new_y = jnp.clip(p2.hook_y + new_vel_y, min_y, max_y)
                final_vel_y = jnp.where((new_y == min_y) | (new_y == max_y), 0.0, new_vel_y)
                return new_y, final_vel_y, p2.hook_state

            p2_new_hook_y, p2_new_hook_velocity_y, p2_hook_state = jax.lax.cond(
                p2.hook_state == 3, p2_auto_lower, p2_normal, operand=None
            )

            # P2 Hook-Position
            p2_hook_x, p2_hook_y = self._get_hook_position_p2(cfg.P2_START_X, PlayerState(
                rod_length=p2_new_rod_length, hook_y=p2_new_hook_y, score=p2.score, hook_state=p2_hook_state,
                hooked_fish_idx=p2.hooked_fish_idx, hook_velocity_y=p2_new_hook_velocity_y,
                hook_x_offset=p2_new_hook_x_offset, display_score=p2.display_score,
                score_animation_timer=p2.score_animation_timer, line_segments_x=p2.line_segments_x
            ))

            # ======== P2: Kollisionslogik / Einhaken ============================================
            can_hook_p2 = (p2_hook_state == 0)
            hook_collides_fish_p2 = (jnp.abs(new_fish_pos[:, 0] - p2_hook_x) < cfg.FISH_WIDTH) & \
                                    (jnp.abs(new_fish_pos[:, 1] - p2_hook_y) < cfg.FISH_HEIGHT)
            valid_targets_p2 = can_hook_p2 & fish_active & hook_collides_fish_p2
            p2_hooked_idx, p2_did_hook = jnp.argmax(valid_targets_p2), jnp.any(valid_targets_p2)
            p2_hook_state = jnp.where(p2_did_hook, 1, p2_hook_state)
            p2_hooked_fish_idx = jnp.where(p2_did_hook, p2_hooked_idx, p2.hooked_fish_idx)
            fish_active = fish_active.at[p2_hooked_idx].set(
                jnp.where(p2_did_hook, False, fish_active[p2_hooked_idx])
            )
            reeling_priority = jnp.where(p2_did_hook & (reeling_priority == -1), 1, reeling_priority)

            # P2 Fast Reel
            is_fire_pressed_p2 = self._is_fire_action(p2_action)
            is_reeling_fast_p2 = (p2_hook_state == 1) & is_fire_pressed_p2

            tick_slow_p2 = (jnp.bitwise_and(state.time, cfg.SLOW_REEL_PERIOD - 1) == 0)
            reel_tick_p2 = jnp.where(is_reeling_fast_p2, True, tick_slow_p2)

            reel_step_p2 = jnp.where(p2_hook_state > 0, 1.0, 0.0)
            can_reel_p2 = p2_hooked_fish_idx >= 0

            p2_new_hook_y = jnp.where(
                (reel_tick_p2 & can_reel_p2),
                jnp.clip(p2_new_hook_y - reel_step_p2, scoring_hook_y, max_hook_y),
                p2_new_hook_y
            )
            # Aktualisierte P2-Hook-Position
            p2_hook_x, p2_hook_y = self._get_hook_position_p2(cfg.P2_START_X, PlayerState(
                rod_length=p2_new_rod_length, hook_y=p2_new_hook_y, score=p2.score, hook_state=p2_hook_state,
                hooked_fish_idx=p2_hooked_fish_idx, hook_velocity_y=p2_new_hook_velocity_y,
                hook_x_offset=p2_new_hook_x_offset, display_score=p2.display_score,
                score_animation_timer=p2.score_animation_timer, line_segments_x=p2.line_segments_x
            ))

            # ======== Wenn P2 eine Fisch hat: Fisch folgt =======================================
            p2_has_hook = (p2_hook_state > 0) & (p2_hooked_fish_idx >= 0)

            def p2_update_hooked():
                fish_idx = p2_hooked_fish_idx
                fish_x, fish_y = new_fish_pos[fish_idx, 0], new_fish_pos[fish_idx, 1]
                fish_dir = new_fish_dirs[fish_idx]
                hx, hy = self._get_hook_position_p2(cfg.P2_START_X, PlayerState(
                    rod_length=p2_new_rod_length, hook_y=p2_new_hook_y, score=0, hook_state=0, hooked_fish_idx=-1,
                    hook_velocity_y=0, hook_x_offset=p2_new_hook_x_offset, display_score=0, score_animation_timer=0,
                    line_segments_x=jnp.zeros(8)
                ))
                depth_ratio = jnp.clip((hy - cfg.WATER_Y_START) / (cfg.MAX_HOOK_DEPTH_Y), 0.0, 1.0)
                wobble_freq = cfg.WOBBLE_FREQ_BASE + depth_ratio * cfg.WOBBLE_FREQ_RANGE
                wobble_amp = cfg.WOBBLE_AMP_BASE + depth_ratio * cfg.WOBBLE_AMP_RANGE
                wobble_dx = jnp.sin(state.time * wobble_freq) * wobble_amp
                base_dx = fish_dir * cfg.FISH_SPEED * cfg.HOOKED_FISH_SPEED_MULTIPLIER
                total_dx = jnp.clip(base_dx + wobble_dx, -cfg.MAX_HOOKED_WOBBLE_DX, cfg.MAX_HOOKED_WOBBLE_DX)
                potential_new_x = fish_x + total_dx
                rod_end_x_local = cfg.P2_START_X - p2_new_rod_length
                effective_right_boundary = cfg.RIGHT_BOUNDARY + 24
                boundary_min = jnp.maximum(rod_end_x_local - cfg.HOOKED_FISH_BOUNDARY_PADDING, cfg.LEFT_BOUNDARY)
                boundary_max = jnp.minimum(rod_end_x_local + cfg.HOOKED_FISH_BOUNDARY_PADDING, effective_right_boundary)

                def apply_hooked_boundaries():
                    would_hit_left = potential_new_x <= boundary_min
                    would_hit_right = potential_new_x >= boundary_max
                    constrained_x = jnp.clip(potential_new_x, boundary_min, boundary_max)
                    new_direction = jnp.where(would_hit_left | would_hit_right, -fish_dir, fish_dir)
                    elastic_push = jnp.where(would_hit_left, 0.5, jnp.where(would_hit_right, -0.5, 0.0))
                    return constrained_x + elastic_push, new_direction

                def apply_global_boundaries():
                    would_hit_left = potential_new_x <= cfg.LEFT_BOUNDARY
                    would_hit_right = potential_new_x >= effective_right_boundary
                    constrained_x = jnp.clip(potential_new_x, cfg.LEFT_BOUNDARY, effective_right_boundary)
                    new_direction = jnp.where(would_hit_left | would_hit_right, -fish_dir, fish_dir)
                    return constrained_x, new_direction

                new_x, new_fish_direction = jax.lax.cond(
                    cfg.HOOKED_FISH_BOUNDARY_ENABLED, apply_hooked_boundaries, apply_global_boundaries
                )
                updated_pos = new_fish_pos.at[fish_idx, 0].set(new_x)
                updated_pos = updated_pos.at[fish_idx, 1].set(hy - cfg.FISH_HEIGHT / 2.0)
                updated_dirs = new_fish_dirs.at[fish_idx].set(new_fish_direction)

                is_facing_right = new_fish_direction > 0
                mouth_x_offset = jnp.where(is_facing_right, cfg.FISH_WIDTH, 0.0)
                hook_target_x = new_x + mouth_x_offset
                rod_end_x_local = cfg.P2_START_X - p2_new_rod_length
                new_offset = hook_target_x - rod_end_x_local

                row_idx = jnp.clip(fish_idx, 0, len(cfg.FISH_PULL_PER_ROW) - 1).astype(jnp.int32)
                fish_pull = jnp.array(cfg.FISH_PULL_PER_ROW)[row_idx]
                tug_key = jax.random.fold_in(state.key, state.time * 37 + 100 + fish_idx)
                do_tug = (jax.random.uniform(tug_key) < (0.08 + 0.04 * depth_ratio)) & (~reel_tick_p2)
                tug_amount = jnp.where(do_tug, fish_pull, 0.0)
                return updated_pos, updated_dirs, new_offset, tug_amount

            tug_amount_p2 = jnp.array(0.0)
            new_fish_pos, new_fish_dirs, p2_new_hook_x_offset, tug_amount_p2 = jax.lax.cond(
                p2_has_hook, lambda _: p2_update_hooked(),
                lambda _: (new_fish_pos, new_fish_dirs, p2_new_hook_x_offset, 0.0),
                operand=None
            )
            p2_new_hook_y = jnp.clip(p2_new_hook_y + tug_amount_p2, 0.0, max_hook_y)

            # ======== P2 Scoring / Shark-Kollision ==============================================
            p2_score = p2.score
            p2_has_hooked_fish = (p2_hook_state > 0) & (p2_hooked_fish_idx >= 0)

            def p2_fpos():
                return new_fish_pos[p2_hooked_fish_idx, 0], new_fish_pos[p2_hooked_fish_idx, 1]

            fish_x_p2, fish_y_p2 = jax.lax.cond(p2_has_hooked_fish, p2_fpos, lambda: (0.0, 0.0))
            fish_center_x_2 = fish_x_p2 + cfg.FISH_WIDTH / 2
            fish_center_y_2 = fish_y_p2 + cfg.FISH_HEIGHT / 2
            collides_x_2 = jnp.abs(fish_center_x_2 - shark_center_x) < (fish_half_w + shark_half_w)
            collides_y_2 = jnp.abs(fish_center_y_2 - shark_center_y) < (fish_half_h + shark_half_h)
            shark_collides_p2 = p2_has_hooked_fish & collides_x_2 & collides_y_2

            p2_hook_at_surface = p2_hook_y <= (cfg.FISH_SCORING_Y + scoring_tolerance)
            p2_fish_at_surface = fish_y_p2 <= (cfg.FISH_SCORING_Y + scoring_tolerance)
            scored_fish_p2 = (p2_hook_state > 0) & (p2_hooked_fish_idx >= 0) & (p2_hook_at_surface | p2_fish_at_surface)
            reset_hook_p2 = shark_collides_p2 | scored_fish_p2

            prev_idx_p2 = p2_hooked_fish_idx
            p2_score += jnp.where(scored_fish_p2, fish_scores[p2_hooked_fish_idx], 0)

            score_increased_p2 = scored_fish_p2
            new_animation_timer_p2 = jnp.where(score_increased_p2, animation_speed,
                                               jnp.where(p2.score_animation_timer > 0, p2.score_animation_timer - 1, 0))
            should_inc_disp_p2 = (new_animation_timer_p2 == 0) & (p2.display_score < p2_score)
            new_display_score_p2 = jnp.where(should_inc_disp_p2, p2.display_score + 1, p2.display_score)
            new_animation_timer_p2 = jnp.where(should_inc_disp_p2 & (new_display_score_p2 < p2_score),
                                               animation_speed, new_animation_timer_p2)

            key, respawn_key_p2 = jax.random.split(key)
            do_respawn_p2 = reset_hook_p2 & (prev_idx_p2 >= 0)
            new_fish_pos, new_fish_dirs = jax.lax.cond(
                do_respawn_p2,
                lambda _: respawn_fish(new_fish_pos, new_fish_dirs, prev_idx_p2, respawn_key_p2),
                lambda _: (new_fish_pos, new_fish_dirs),
                operand=None
            )
            p2_hook_state = jnp.where(reset_hook_p2, 3, p2_hook_state)
            p2_hooked_fish_idx = jnp.where(reset_hook_p2, -1, p2_hooked_fish_idx)
            fish_active = jnp.where(do_respawn_p2, fish_active.at[prev_idx_p2].set(True), fish_active)

            # ======== Game Over =================================================================
            game_over = (p1_score >= 99) | (p2_score >= 99)

            # ======== Neuen State zusammenbauen =================================================
            return GameState(
                p1=PlayerState(
                    rod_length=new_rod_length,
                    hook_y=new_hook_y,
                    score=p1_score,
                    hook_state=p1_hook_state,
                    hooked_fish_idx=p1_hooked_fish_idx,
                    hook_velocity_y=new_hook_velocity_y,
                    hook_x_offset=new_hook_x_offset,
                    display_score=new_display_score_p1,
                    score_animation_timer=new_animation_timer_p1,
                    line_segments_x=p1.line_segments_x
                ),
                p2=PlayerState(
                    rod_length=p2_new_rod_length,
                    hook_y=p2_new_hook_y,
                    score=p2_score,
                    hook_state=p2_hook_state,
                    hooked_fish_idx=p2_hooked_fish_idx,
                    hook_velocity_y=p2_new_hook_velocity_y,
                    hook_x_offset=p2_new_hook_x_offset,
                    display_score=new_display_score_p2,
                    score_animation_timer=new_animation_timer_p2,
                    line_segments_x=p2.line_segments_x
                ),
                fish_positions=new_fish_pos,
                fish_directions=new_fish_dirs,
                fish_active=fish_active,
                fish_turn_cooldowns=new_cooldowns,
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
        constant_values=0
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
        'fish2': os.path.join(MODULE_DIR, "sprites/fishingderby/fish4.npy"),
        'sky': os.path.join(MODULE_DIR, "sprites/fishingderby/sky.npy"),
        'pier': os.path.join(MODULE_DIR, "sprites/fishingderby/pier.npy"),
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
        sprites['sky'], score_digits, sprites['pier']
    )


class FishingDerbyRenderer(JAXGameRenderer):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        (
            self.SPRITE_BG, self.SPRITE_PLAYER1, self.SPRITE_PLAYER2,
            self.SPRITE_SHARK1, self.SPRITE_SHARK2, self.SPRITE_FISH1,
            self.SPRITE_FISH2, self.SPRITE_SKY, self.SPRITE_SCORE_DIGITS, self.SPRITE_PIER
        ) = load_sprites()

    def _get_hook_position(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        """Calculate the actual hook position based on rod length and hook depth."""
        cfg = self.config
        rod_end_x = player_x + player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    def _get_hook_position_p2(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        cfg = self.config
        # Player 2's rod extends leftward, so subtract rod length from starting position
        rod_end_x = player_x - player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> chex.Array:
        cfg = self.config
        raster = jnp.zeros((cfg.SCREEN_HEIGHT, cfg.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Draw sky
        raster = raster.at[:cfg.WATER_Y_START, :, :].set(jnp.array(cfg.SKY_COLOR, dtype=jnp.uint8))

        # Draw base water color first
        base_water = jnp.array(cfg.WATER_COLOR, dtype=jnp.uint8)
        raster = raster.at[cfg.WATER_Y_START:, :, :].set(base_water)

        # Draw background shimmer (top 10 pixels of shimmer region)
        SHIMMER_HEIGHT = 16
        BACKGROUND_SHIMMER_HEIGHT = 10
        shimmer_start = cfg.WATER_Y_START
        background_shimmer_end = shimmer_start + BACKGROUND_SHIMMER_HEIGHT

        # Generate background shimmer
        shimmer_time = state.time // 8
        y_indices_bg = jnp.arange(BACKGROUND_SHIMMER_HEIGHT)

        def compute_shimmer_color(y_offset):
            y = shimmer_start + y_offset
            color_hash = (shimmer_time * 17 + y * 11) & 255
            use_light = (color_hash & 3) > 1

            light_color = jnp.array([
                base_water[0] + 6,
                base_water[1] + 8,
                base_water[2] + 6
            ], dtype=jnp.int16)

            dark_color = jnp.array([
                base_water[0] - 4,
                base_water[1] - 4,
                base_water[2] - 2
            ], dtype=jnp.int16)

            shimmer_color = jnp.where(use_light, light_color, dark_color)
            return jnp.clip(shimmer_color, 0, 255).astype(jnp.uint8)

        # Apply background shimmer
        def apply_background_shimmer(carry, y_offset):
            r = carry
            y = shimmer_start + y_offset
            shimmer_color = compute_shimmer_color(y_offset)
            r = r.at[y, :, :].set(shimmer_color)
            return r, None

        raster, _ = jax.lax.scan(apply_background_shimmer, raster, y_indices_bg)



        # Draw players
        raster = self._render_at(raster, cfg.P1_START_X, cfg.PLAYER_Y, self.SPRITE_PLAYER1)
        raster = self._render_at(raster, cfg.P2_START_X, cfg.PLAYER_Y, self.SPRITE_PLAYER2)

        # Draw Player 1 fishing line
        p1_rod_end_x = cfg.P1_START_X + state.p1.rod_length
        hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, state.p1)

        # Draw horizontal part of Player 1 rod
        raster = self._render_line(raster, cfg.P1_START_X + 7, cfg.ROD_Y, p1_rod_end_x, cfg.ROD_Y, (0, 0, 0))

        # Player 1 line rendering logic
        in_water = state.p1.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
        is_reeling = state.p1.hook_state > 0
        has_horizontal_offset = jnp.abs(state.p1.hook_x_offset) > 0.5
        apply_sag = in_water & ~is_reeling & has_horizontal_offset

        water_depth_ratio = jnp.clip((state.p1.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0,
                                     1.0)
        sag_amount = jnp.where(apply_sag, 3.0 + water_depth_ratio * 3.0, 0.0)

        line_start = jnp.array([p1_rod_end_x, cfg.ROD_Y])
        line_end = jnp.array([hook_x, hook_y])

        raster = jax.lax.cond(
            apply_sag,
            lambda r: self._render_saggy_line(r, line_start, line_end, sag_amount, (255, 255, 0)),
            lambda r: self._render_line(r, line_start[0], line_start[1], line_end[0], line_end[1], (255, 255, 0)),
            raster
        )

        # Draw Player 2 fishing line
        p2_rod_end_x = cfg.P2_START_X - state.p2.rod_length
        p2_hook_x, p2_hook_y = self._get_hook_position_p2(cfg.P2_START_X, state.p2)

        # Draw horizontal part of Player 2 rod (extends leftward)
        raster = self._render_line(raster, cfg.P2_START_X + 8, cfg.ROD_Y, p2_rod_end_x, cfg.ROD_Y, (0, 0, 0))

        # Player 2 line rendering logic
        p2_in_water = state.p2.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
        p2_is_reeling = state.p2.hook_state > 0
        p2_has_horizontal_offset = jnp.abs(state.p2.hook_x_offset) > 0.5
        p2_apply_sag = p2_in_water & ~p2_is_reeling & p2_has_horizontal_offset
        p2_water_depth_ratio = jnp.clip((state.p2.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0,
                                        1.0)
        p2_sag_amount = jnp.where(p2_apply_sag, 3.0 + p2_water_depth_ratio * 3.0, 0.0)
        p2_line_start = jnp.array([p2_rod_end_x, cfg.ROD_Y])
        p2_line_end = jnp.array([p2_hook_x, p2_hook_y])

        raster = jax.lax.cond(
            p2_apply_sag,
            lambda r: self._render_saggy_line(r, p2_line_start, p2_line_end, p2_sag_amount, (0, 0, 0)),
            lambda r: self._render_line(r, p2_line_start[0], p2_line_start[1], p2_line_end[0], p2_line_end[1],
                                        (0, 0, 0)),
            raster
        )

        # Draw shark
        shark_frame = jax.lax.cond((state.time // 4) % 2 == 0, lambda: self.SPRITE_SHARK1, lambda: self.SPRITE_SHARK2)
        raster = self._render_at(raster, state.shark_x, cfg.SHARK_Y, shark_frame, flip_h=state.shark_dir < 0)

        # Draw fish - fix sprite flipping to ensure hook appears at mouth
        fish_frame = jax.lax.cond((state.time // 5) % 2 == 0, lambda: self.SPRITE_FISH1, lambda: self.SPRITE_FISH2)

        def draw_one_fish(i, r):
            pos, direction, active = state.fish_positions[i], state.fish_directions[i], state.fish_active[i]
            is_hooked_p1 = (state.p1.hooked_fish_idx == i) & (state.p1.hook_state > 0)
            is_hooked_p2 = (state.p2.hooked_fish_idx == i) & (state.p2.hook_state > 0)
            is_hooked = is_hooked_p1 | is_hooked_p2

            hooked_fish_frame = jax.lax.cond((state.time // 6) % 2 == 0, lambda: self.SPRITE_FISH1,
                                             lambda: self.SPRITE_FISH2)
            frame_to_use = jax.lax.cond(is_hooked, lambda: hooked_fish_frame, lambda: fish_frame)

            # Fix flipping: fish should face direction of travel, flip when direction > 0 (moving right)
            flip_sprite = direction > 0

            return jax.lax.cond(active,
                                lambda r_in: self._render_at(r_in, pos[0], pos[1], frame_to_use, flip_h=flip_sprite),
                                lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, cfg.NUM_FISH, draw_one_fish, raster)

        # Draw hooked fish for Player 1
        def draw_hooked_p1(r):
            fish_idx = state.p1.hooked_fish_idx
            fish_pos = state.fish_positions[fish_idx]
            fish_dir = state.fish_directions[fish_idx]

            # Fix flipping: fish should face direction of travel, flip when direction > 0 (moving right)
            flip_sprite = fish_dir > 0

            fish_frame = jax.lax.cond((state.time // 2) % 2 == 0, lambda: self.SPRITE_FISH1, lambda: self.SPRITE_FISH2)
            return self._render_at(r, fish_pos[0], fish_pos[1], fish_frame, flip_h=flip_sprite)

        should_draw_hooked_p1 = (state.p1.hook_state > 0) & (state.p1.hooked_fish_idx >= 0) & (state.p1.hook_state != 3)
        raster = jax.lax.cond(should_draw_hooked_p1, draw_hooked_p1, lambda r: r, raster)

        # Draw hooked fish for Player 2
        def draw_hooked_p2(r):
            fish_idx = state.p2.hooked_fish_idx
            fish_pos = state.fish_positions[fish_idx]
            fish_dir = state.fish_directions[fish_idx]

            # Fix flipping: fish should face direction of travel, flip when direction > 0 (moving right)
            flip_sprite = fish_dir > 0

            fish_frame = jax.lax.cond((state.time // 2) % 2 == 0, lambda: self.SPRITE_FISH1, lambda: self.SPRITE_FISH2)
            return self._render_at(r, fish_pos[0], fish_pos[1], fish_frame, flip_h=flip_sprite)

        should_draw_hooked_p2 = (state.p2.hook_state > 0) & (state.p2.hooked_fish_idx >= 0) & (state.p2.hook_state != 3)
        raster = jax.lax.cond(should_draw_hooked_p2, draw_hooked_p2, lambda r: r, raster)

        # Draw hooks
        raster = self._render_at(raster, hook_x - cfg.HOOK_WIDTH // 2, hook_y - cfg.HOOK_HEIGHT // 2,
                                 jnp.array([[[255, 255, 255]]], dtype=jnp.uint8))
        raster = self._render_at(raster, p2_hook_x - cfg.HOOK_WIDTH // 2, p2_hook_y - cfg.HOOK_HEIGHT // 2,
                                 jnp.array([[[255, 255, 255]]], dtype=jnp.uint8))
        # Draw pier on top
        raster = self._render_at(raster, 0, cfg.WATER_Y_START - 10, self.SPRITE_PIER)
        # Draw scores
        raster = self._render_score(raster, state.p1.display_score, 50, 10)
        raster = self._render_score(raster, state.p2.display_score, 100, 10)

        # This renders ABOVE all sprites for realistic water surface effect
        FOREGROUND_SHIMMER_HEIGHT = 6
        foreground_shimmer_start = background_shimmer_end
        foreground_shimmer_end = shimmer_start + SHIMMER_HEIGHT

        y_indices_fg = jnp.arange(FOREGROUND_SHIMMER_HEIGHT)

        def apply_foreground_shimmer(carry, y_offset):
            r = carry
            y = foreground_shimmer_start + y_offset
            # Use same shimmer calculation but with different y offset
            shimmer_color = compute_shimmer_color(BACKGROUND_SHIMMER_HEIGHT + y_offset)

            # Apply shimmer with some transparency effect by blending with existing pixels
            existing_color = r[y, :, :]
            blended_color = (shimmer_color.astype(jnp.int16) + existing_color.astype(jnp.int16)) // 2
            blended_color = jnp.clip(blended_color, 0, 255).astype(jnp.uint8)

            r = r.at[y, :, :].set(blended_color)
            return r, None

        raster, _ = jax.lax.scan(apply_foreground_shimmer, raster, y_indices_fg)

        return raster

    def _render_score(self, raster, display_score, x, y):
        s1, s0 = display_score // 10, display_score % 10
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

    # Simplified input detection that allows diagonal movement
    # No exclusion between directions, allowing simultaneous presses
    is_up = up
    is_down = down
    is_left = left
    is_right = right

    # Prevent conflicting directions (up+down or left+right)
    # If both opposing directions are pressed, neither takes effect
    if is_up and is_down:
        is_up = is_down = False
    if is_left and is_right:
        is_left = is_right = False

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
