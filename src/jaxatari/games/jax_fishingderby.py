import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, List, Optional

import chex
import jax
import jax.numpy as jnp

import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Create procedural background with sky and water colors
    # Using default values from GameConfig
    SCREEN_HEIGHT = 210
    SCREEN_WIDTH = 160
    WATER_Y_START = 60
    SKY_COLOR = (117, 128, 240)  # RGB
    WATER_COLOR = (24, 26, 167)  # RGB
    
    # Create RGBA background
    background = jnp.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=jnp.uint8)
    
    # Set sky color (top part)
    sky_rgb = jnp.array(SKY_COLOR + (255,), dtype=jnp.uint8)  # Add alpha
    background = background.at[:WATER_Y_START, :, :].set(sky_rgb)
    
    # Set water color (bottom part)
    water_rgb = jnp.array(WATER_COLOR + (255,), dtype=jnp.uint8)  # Add alpha
    background = background.at[WATER_Y_START:, :, :].set(water_rgb)
    
    return {
        'background': background,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Fishing Derby.
    """
    procedural_sprites = _create_static_procedural_sprites()
    asset_list = [
        {'name': 'background', 'type': 'background', 'data': procedural_sprites['background']},
        {'name': 'player1', 'type': 'single', 'file': 'player1.npy'},
        {'name': 'player2', 'type': 'single', 'file': 'player2.npy'},
        {'name': 'sky', 'type': 'single', 'file': 'sky.npy'},
        {'name': 'pier', 'type': 'single', 'file': 'pier.npy'},
        {'name': 'shark', 'type': 'group', 'files': ['shark_new_1.npy', 'shark_new_2.npy']},
        {'name': 'fish', 'type': 'group', 'files': ['fish1.npy', 'fish4.npy']},
        {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        {
            'name': 'line_colors',
            'type': 'procedural',
            'data': jnp.array(
                [
                    [[0, 0, 0, 255]],
                    [[255, 255, 0, 255]],
                    [[255, 255, 255, 255]],
                ],
                dtype=jnp.uint8,
            ).reshape(3, 1, 1, 4),
        },
    ]
    return tuple(asset_list)


class GameConfig(NamedTuple):
    """All static configuration parameters for the game."""
    # Screen dimensions
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    
    # Colors
    SKY_COLOR: Tuple[int, int, int] = (117, 128, 240)
    WATER_COLOR: Tuple[int, int, int] = (24, 26, 167)
    
    # Water
    WATER_Y_START: int = 60
    WATER_SHIMMER_HEIGHT: int = 16
    
    # Game control
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

    # Occasional downward tugs by row (px/frame) when you're NOT reeling on this frame
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
    
    # Asset configuration
    ASSET_CONFIG: tuple = None  # Will be set via factory method
    
    @classmethod
    def create_default(cls):
        """Factory method to create GameConfig with default asset config."""
        # Create with all defaults, using _replace to set ASSET_CONFIG
        # We need to provide all fields, so we'll use the defaults from the class definition
        return cls(
            SCREEN_WIDTH=160,
            SCREEN_HEIGHT=210,
            SKY_COLOR=(117, 128, 240),
            WATER_COLOR=(24, 26, 167),
            WATER_Y_START=60,
            WATER_SHIMMER_HEIGHT=16,
            RESET=18,
            P1_START_X=9,
            P2_START_X=135,
            PLAYER_Y=23,
            ROD_Y=38,
            FISH_SCORING_Y=78,
            MIN_ROD_LENGTH_X=23,
            START_ROD_LENGTH_X=23,
            MAX_ROD_LENGTH_X=65,
            P2_MIN_ROD_LENGTH_X=7,
            P2_MAX_ROD_LENGTH_X=46,
            MIN_HOOK_DEPTH_Y=0,
            START_HOOK_DEPTH_Y=40,
            MAX_HOOK_DEPTH_Y=160,
            ROD_SPEED=1.0,
            FISH_DEATH_LINE_OFFSET=20,
            HOOK_WIDTH=3,
            HOOK_HEIGHT=5,
            HOOK_SPEED_V=10.0,
            REEL_SLOW_SPEED=1.0,
            LINE_Y_START=48,
            LINE_Y_END=180,
            AUTO_LOWER_SPEED=2.0,
            Acceleration=0.2,
            Damping=0.85,
            SLOW_REEL_PERIOD=6,
            MAX_HOOKED_WOBBLE_DX=0.9,
            WOBBLE_FREQ_BASE=0.10,
            WOBBLE_FREQ_RANGE=0.06,
            WOBBLE_AMP_BASE=0.05,
            WOBBLE_AMP_RANGE=0.20,
            FISH_PULL_PER_ROW=(0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
            LEFT_BOUNDARY=10.0,
            RIGHT_BOUNDARY=115.0,
            FISH_WIDTH=8,
            FISH_HEIGHT=7,
            FISH_SPEED=0.4,
            NUM_FISH=6,
            FISH_ROW_YS=(95, 111, 127, 143, 159, 175),
            FISH_ROW_SCORES=(2, 2, 4, 4, 6, 6),
            HOOKED_FISH_SPEED_MULTIPLIER=1.5,
            HOOKED_FISH_TURN_PROBABILITY=0.04,
            HOOKED_FISH_BOUNDARY_ENABLED=True,
            HOOKED_FISH_BOUNDARY_PADDING=20,
            FISH_BASE_TURN_PROBABILITY=0.01,
            HOOKED_FISH_TURNING_COOLDOWN=30,
            SHARK_WIDTH=16,
            SHARK_HEIGHT=7,
            SHARK_SPEED=0.3,
            SHARK_Y=78,
            SHARK_BURST_SPEED=1.5,
            SHARK_BURST_DURATION=300,
            SHARK_BURST_CHANCE=0.001,
            ASSET_CONFIG=_get_default_asset_config()
        )

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
    fish_turn_cooldowns: chex.Array
    shark_x: chex.Array
    shark_dir: chex.Array
    shark_burst_timer: chex.Array
    reeling_priority: chex.Array
    time: chex.Array
    game_over: chex.Array
    key: jax.random.PRNGKey

"""
    Represents the observation of the game state at a specific moment.

    Attributes:
        player1_hook_xy (chex.Array): The x and y coordinates of Player 1's hook.
        fish_xy (chex.Array): The x and y coordinates of all fish in the game.
        shark_x (chex.Array): The x coordinate of the shark.
        score (chex.Array): The current score of Player 1.
"""
class FishingDerbyObservation(NamedTuple):
    player1_hook_xy: chex.Array
    fish_xy: chex.Array
    shark_x: chex.Array
    score: chex.Array

"""
    Represents additional information about the current game state.

    Attributes:
        p1_score (int): The current score of Player 1.
        p2_score (int): The current score of Player 2.
        time (int): The elapsed time in the game, measured in frames.
"""
class FishingDerbyInfo(NamedTuple):
    p1_score: int
    p2_score: int
    time: int


# Game Logic
"""
    Represents the Fishing Derby game environment.

    This class extends the `JaxEnvironment` base class and implements the core
    game logic, including player actions, fish and shark movement, scoring, and
    collision detection.
"""
class FishingDerby(JaxEnvironment):
    def __init__(self, consts: Optional[GameConfig] = None):
        consts = consts or GameConfig.create_default()
        super().__init__()
        self.consts = consts
        self.renderer = FishingDerbyRenderer(self.consts)
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
        cfg = self.consts
        rod_end_x = player_x + player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    def _get_hook_position_p2(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        """Calculate the actual hook position for Player 2 based on rod length and hook depth."""
        cfg = self.consts
        # Player 2's rod extends leftward, so subtract rod length from starting position
        rod_end_x = player_x - player_state.rod_length
        # Apply horizontal offset for water resistance effect
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(10)) -> Tuple[FishingDerbyObservation, GameState]:
        """
                Resets the Fishing Derby game environment to its initial state.

                This method initializes the game state, including player positions, fish positions,
                and other game-related variables. It also generates a new random key for reproducibility.

                Args:
                    key (jax.random.PRNGKey, optional): A JAX random key used for initializing
                        random elements in the game. Defaults to `jax.random.PRNGKey(10)`.

                Returns:
                    Tuple[FishingDerbyObservation, GameState]: A tuple containing the initial observation
                        of the game state and the full game state.
                """
        key, fish_key = jax.random.split(key)

        p1_state = PlayerState(
            rod_length=jnp.array(float(self.consts.START_ROD_LENGTH_X)),
            hook_y=jnp.array(float(self.consts.START_HOOK_DEPTH_Y)),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0),
            hook_x_offset=jnp.array(0.0),
            display_score=jnp.array(0),
            score_animation_timer=jnp.array(0),
            line_segments_x=jnp.zeros(8)
        )

        p2_state = PlayerState(
            rod_length=jnp.array(float(self.consts.START_ROD_LENGTH_X)),
            hook_y=jnp.array(float(self.consts.WATER_Y_START - self.consts.ROD_Y)),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0),
            hook_x_offset=jnp.array(0.0),
            display_score=jnp.array(0),
            score_animation_timer=jnp.array(0),
            line_segments_x=jnp.zeros(8)
        )

        fish_x = jax.random.uniform(fish_key, (self.consts.NUM_FISH,), minval=self.consts.LEFT_BOUNDARY,
                                    maxval=self.consts.RIGHT_BOUNDARY)
        fish_y = jnp.array(self.consts.FISH_ROW_YS, dtype=jnp.float32)

        state = GameState(
            p1=p1_state, p2=p2_state,
            fish_positions=jnp.stack([fish_x, fish_y], axis=1),
            fish_directions=jax.random.choice(key, jnp.array([-1.0, 1.0]), (self.consts.NUM_FISH,)),
            fish_active=jnp.ones(self.consts.NUM_FISH, dtype=jnp.bool_),
            fish_turn_cooldowns=jnp.zeros(self.consts.NUM_FISH, dtype=jnp.int32),  # Initialize cooldowns
            shark_x=jnp.array(self.consts.SCREEN_WIDTH / 2.0),
            shark_dir=jnp.array(1.0),
            shark_burst_timer=jnp.array(0),
            reeling_priority=jnp.array(-1),
            time=jnp.array(0),
            game_over=jnp.array(False),
            key=key
        )
        return self._get_observation(state), state

    def _get_observation(self, state: GameState) -> FishingDerbyObservation:
        """
            Generates an observation of the current game state for Player 1.

            This method calculates the hook position for Player 1 and compiles
            relevant game state information into a `FishingDerbyObservation` object.

            Args:
                state (GameState): The current state of the game.

            Returns:
                FishingDerbyObservation: An object containing the x and y coordinates
                of Player 1's hook, the positions of all fish, the x coordinate of the shark,
                and Player 1's current score.
            """
        hook_x, hook_y = self._get_hook_position(self.consts.P1_START_X, state.p1)
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

    def _get_done(self, state: GameState) -> bool:
        """
            Determines whether the game is over.

            Args:
                state (GameState): The current state of the game.

            Returns:
                bool: True if the game is over, False otherwise.
            """
        return state.game_over

    def _get_info(self, state: GameState) -> FishingDerbyInfo:
        """
           Retrieves additional information about the current game state.

           Args:
               state (GameState): The current state of the game.

           Returns:
               FishingDerbyInfo: An object containing Player 1's score, Player 2's score,
               the elapsed time in the game, and rewards for all players.
           """
        return FishingDerbyInfo(
            p1_score=state.p1.score,
            p2_score=state.p2.score,
            time=state.time,
        )


    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int, p2_action: int = -1) -> Tuple[
        FishingDerbyObservation, GameState, chex.Array, bool, FishingDerbyInfo]:
        """Processes one frame of the game and returns the full tuple."""

        key, p2_key = jax.random.split(state.key)

        def strategic_p2_ai():
            """
            Strategy:
            - Try to target bottom 3 fish rows (worth 4-6 points each)
            - Fast reels when shark is on the left side
            - Properly retracts rod when reeling in fish

            """
            cfg = self.consts
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



                # If we're in the catch zone, just go down slowly to hook
                action_in_zone = jnp.where(
                    dy > 2,  # If slightly below, go down
                    Action.DOWN,
                    Action.NOOP  # Otherwise wait for fish
                )

                # Horizontal + Vertical combinations
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

                # Decision priority
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
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        """
            Returns the action space of the environment.

            This method defines the discrete action space available to the players
            based on the number of actions in the `action_set`.

            Returns:
                spaces.Discrete: A discrete space object representing the number of possible actions.
            """
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        """
            Retrieves the action set as a JAX array.

            This method converts the list of possible actions (`action_set`) into
            a JAX array for efficient computation.

            Returns:
                jnp.ndarray: A JAX array containing all possible actions.
            """
        return jnp.array(self.action_set)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space of the environment."""
        return spaces.Dict({
            "player1_hook_xy": spaces.Box(
                low=jnp.array([0.0, 0.0], dtype=jnp.float32),
                high=jnp.array([self.consts.SCREEN_WIDTH, self.consts.SCREEN_HEIGHT], dtype=jnp.float32),
                shape=(2,),
                dtype=jnp.float32
            ),
            "fish_xy": spaces.Box(
                low=jnp.array([[0.0, 0.0]] * self.consts.NUM_FISH, dtype=jnp.float32),
                high=jnp.array([[self.consts.SCREEN_WIDTH, self.consts.SCREEN_HEIGHT]] * self.consts.NUM_FISH,
                               dtype=jnp.float32),
                shape=(self.consts.NUM_FISH, 2),
                dtype=jnp.float32
            ),
            "shark_x": spaces.Box(
                low=jnp.array(0.0, dtype=jnp.float32),
                high=jnp.array(self.consts.SCREEN_WIDTH, dtype=jnp.float32),
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
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
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
        cfg = self.consts

        def reset_branch(_):
            _, new_state = self.reset(state.key)
            return new_state

        def game_branch(_):
            # Shorthand
            p1 = state.p1
            p2 = state.p2

            # RNG
            key = state.key
            key, fish_key = jax.random.split(key)

            # Fish movement
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

            # shark movement
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

           # shared calculations

            min_hook_y = 0.0
            max_hook_y = cfg.LINE_Y_END - cfg.ROD_Y
            water_surface_hook_y = float(cfg.WATER_Y_START - cfg.ROD_Y)

            scoring_hook_y = float(cfg.FISH_SCORING_Y - cfg.ROD_Y)

            # P1 rod horizontal
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

            # P2: Rod horizontal (mirrored logic)
            p2_rod_change = 0.0
            p2_rod_change = jnp.where(p2_action == Action.LEFT, +cfg.ROD_SPEED, p2_rod_change)
            p2_rod_change = jnp.where(p2_action == Action.RIGHT, -cfg.ROD_SPEED, p2_rod_change)
            p2_new_rod_length = jnp.clip(p2.rod_length + p2_rod_change, cfg.P2_MIN_ROD_LENGTH_X,cfg.P2_MAX_ROD_LENGTH_X)

            # P1 water resistance Hook-X-Offset
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
                """
                    Applies water resistance to Player 1's hook movement.

                    This function calculates the horizontal offset of Player 1's hook due to water resistance.
                    The offset is influenced by the depth of the hook, the movement of the rod, and the current
                    offset. It also accounts for directional lag when the rod is moving.

                    Returns:
                        float: The new horizontal offset for Player 1's hook.
                    """
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
                """
                    Applies air recovery to Player 1's hook movement.

                    This function reduces the horizontal offset of Player 1's hook when it is
                    outside the water. The reduction is proportional to the air recovery factor.

                    Returns:
                        float: The updated horizontal offset for Player 1's hook.
                    """
                return p1.hook_x_offset * (1.0 - air_recovery_factor)

            new_hook_x_offset = jax.lax.cond(p1_in_water, p1_apply_water_resistance, p1_apply_air_recovery)

            # P2 water resistance Hook-X-Offset (mirrored logic)
            p2_in_water = p2.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
            rod_end_x_p2 = cfg.P2_START_X - p2_new_rod_length
            target_hook_x_p2 = rod_end_x_p2

            current_hook_x_p2 = cfg.P2_START_X - p2.rod_length + p2.hook_x_offset
            actual_rod_change_p2 = p2_new_rod_length - p2.rod_length
            depth_factor_p2 = jnp.clip((p2.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0, 1.0)
            resistance_multiplier_p2 = 1.0 + depth_factor_p2 * 2.0

            def p2_apply_water_resistance():
                """
                    Applies water resistance to Player 2's hook movement.

                    This function calculates the horizontal offset of Player 2's hook due to water resistance.
                    The offset is influenced by the depth of the hook, the movement of the rod, and the current
                    offset. It also accounts for directional lag when the rod is moving.

                    Returns:
                        float: The new horizontal offset for Player 2's hook.
                    """
                resistance = water_resistance_factor * resistance_multiplier_p2
                target_offset = target_hook_x_p2 - rod_end_x_p2
                current_offset = p2.hook_x_offset
                is_moving = jnp.abs(actual_rod_change_p2) > 0.01
                moving_offset = current_offset + (target_offset - current_offset) * resistance

                rod_moving_left = actual_rod_change_p2 > 0
                rod_moving_right = actual_rod_change_p2 < 0
                lag_magnitude = jnp.abs(actual_rod_change_p2) * 0.8 * resistance_multiplier_p2
                directional_lag = jnp.where(rod_moving_right, -lag_magnitude,
                                            jnp.where(rod_moving_left, +lag_magnitude, 0.0))
                moving_result = moving_offset + directional_lag
                stationary_result = current_offset * (1.0 - smooth_recovery_factor)
                return jnp.where(is_moving, moving_result, stationary_result)

            def p2_apply_air_recovery():
                """
                    Applies air recovery to Player 2's hook movement.

                    This function reduces the horizontal offset of Player 2's hook when it is
                    outside the water. The reduction is proportional to the air recovery factor.

                    Returns:
                        float: The updated horizontal offset for Player 2's hook.
                    """
                return p2.hook_x_offset * (1.0 - air_recovery_factor)

            p2_new_hook_x_offset = jax.lax.cond(p2_in_water, p2_apply_water_resistance, p2_apply_air_recovery)

            # P1 Hook Vertical Movement and State Machine
            def p1_auto_lower(_):
                """
                    Automatically lowers Player 1's hook until it reaches the water surface.

                    This function increments the vertical position of the hook by the auto-lower speed
                    and resets the hook's velocity. If the hook reaches the water surface, its state is
                    updated to indicate that it is no longer in the auto-lowering phase.

                    Args:
                        _ (Any): Placeholder argument, not used in the function.

                    Returns:
                        Tuple[float, float, int]: A tuple containing:
                            - The updated vertical position of the hook.
                            - The updated vertical velocity of the hook (always 0.0).
                            - The updated hook state (0 if the hook reaches the water surface, otherwise unchanged).
                    """
                new_y = p1.hook_y + cfg.AUTO_LOWER_SPEED
                new_vel_y = 0.0
                hook_reached_water = new_y >= water_surface_hook_y
                final_state = jnp.where(hook_reached_water, 0, p1.hook_state)
                final_y = jnp.where(hook_reached_water, water_surface_hook_y, new_y)
                return final_y, new_vel_y, final_state

            def p1_normal(_):
                """
                    Handles the normal vertical movement of Player 1's hook.

                    This function calculates the vertical velocity and position of the hook
                    based on the player's input actions. It ensures that the hook's movement
                    respects the acceleration, damping, and boundary constraints. The hook
                    can only move vertically when it is in the free state (hook_state == 0).

                    Args:
                        _ (Any): Placeholder argument, not used in the function.

                    Returns:
                        Tuple[float, float, int]: A tuple containing:
                            - The updated vertical position of the hook.
                            - The updated vertical velocity of the hook.
                            - The current hook state (unchanged in this function).
                    """
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

            # P1 Hook Position
            hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, PlayerState(
                rod_length=new_rod_length, hook_y=new_hook_y, score=p1.score, hook_state=p1_hook_state,
                hooked_fish_idx=p1.hooked_fish_idx, hook_velocity_y=new_hook_velocity_y,
                hook_x_offset=new_hook_x_offset,
                display_score=p1.display_score, score_animation_timer=p1.score_animation_timer,
                line_segments_x=p1.line_segments_x
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
            is_fire_pressed_p1 = self._is_fire_action(p1_action)
            is_reeling_fast_p1 = (p1_hook_state == 1) & is_fire_pressed_p1

            # Fast reel is every frame, slow reel is every cfg.SLOW_REEL_PERIOD frames
            tick_slow = (jnp.bitwise_and(state.time, cfg.SLOW_REEL_PERIOD - 3) == 0)
            reel_tick = jnp.where(is_reeling_fast_p1, True, tick_slow)

            reel_step = jnp.where(p1_hook_state > 0, 1.0, 0.0)  # 1 px per tick
            can_reel = p1_hooked_fish_idx >= 0

            new_hook_y = jnp.where(
                (reel_tick & can_reel),
                jnp.clip(new_hook_y - reel_step, scoring_hook_y, max_hook_y),
                new_hook_y
            )

            # If P1 reaches scoring depth with a hooked fish, score it
            has_hook = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0)

            def p1_update_hooked():
                """
                    Updates the position and behavior of a hooked fish for Player 1.

                    This function calculates the new position of the hooked fish based on its
                    movement direction, wobble effect, and boundaries. It also ensures that the
                    fish stays within the allowed boundaries relative to the rod's position.

                    Variables:
                        fish_idx (int): The index of the hooked fish.
                        fish_x (float): The current x-coordinate of the hooked fish.
                        fish_y (float): The current y-coordinate of the hooked fish.
                        fish_dir (float): The current movement direction of the hooked fish (-1 for left, 1 for right).
                        hx (float): The x-coordinate of the hook.
                        hy (float): The y-coordinate of the hook.
                        depth_ratio (float): The normalized depth of the hook in the water.
                        wobble_freq (float): The frequency of the wobble effect based on the hook's depth.
                        wobble_amp (float): The amplitude of the wobble effect based on the hook's depth.
                        wobble_dx (float): The horizontal displacement caused by the wobble effect.
                        base_dx (float): The base horizontal movement of the fish.
                        total_dx (float): The total horizontal movement of the fish, including wobble.
                        potential_new_x (float): The potential new x-coordinate of the fish.
                        rod_end_x_local (float): The x-coordinate of the rod's end.
                        boundary_min (float): The minimum x-boundary for the fish's movement.
                        boundary_max (float): The maximum x-boundary for the fish's movement.
                    """
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
                    """
                        Applies boundaries for a hooked fish's movement.

                        This function ensures that the hooked fish stays within the allowed boundaries
                        relative to the rod's position. If the fish hits the boundary, its direction is
                        reversed, and an elastic push effect is applied to simulate a bounce.

                        Returns:
                            Tuple[float, float]: A tuple containing:
                                - The constrained x-coordinate of the fish after applying boundaries.
                                - The updated movement direction of the fish.
                        """
                    would_hit_left = potential_new_x <= boundary_min
                    would_hit_right = potential_new_x >= boundary_max
                    constrained_x = jnp.clip(potential_new_x, boundary_min, boundary_max)
                    new_direction = jnp.where(would_hit_left | would_hit_right, -fish_dir, fish_dir)
                    elastic_push = jnp.where(would_hit_left, 0.5, jnp.where(would_hit_right, -0.5, 0.0))
                    return constrained_x + elastic_push, new_direction

                def apply_global_boundaries():
                    """
                        Ensures that a fish's movement stays within the global boundaries.

                        This function checks if the fish's potential new x-coordinate exceeds the
                        global left or right boundaries. If so, it constrains the x-coordinate to
                        the boundary and reverses the fish's direction.

                        Returns:
                            Tuple[float, float]: A tuple containing:
                                - The constrained x-coordinate of the fish.
                                - The updated movement direction of the fish.
                        """
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

            # P1 Scoring, Shark Collision, and Respawn Logic
            p1_score = p1.score
            has_hooked_fish = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0)

            def p1_fpos():
                return new_fish_pos[p1_hooked_fish_idx, 0], new_fish_pos[p1_hooked_fish_idx, 1]

            fish_x_p1, fish_y_p1 = jax.lax.cond(has_hooked_fish, p1_fpos, lambda: (0.0, 0.0))

            # Shark collision logic
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
                """
                    Respawns a fish at a new position and direction.

                    This function generates a new random position and direction for a fish
                    that needs to be respawned. The position is determined based on the
                    fish's row index, and the direction is chosen randomly.

                    Args:
                        all_pos (chex.Array): The current positions of all fish.
                        all_dirs (chex.Array): The current directions of all fish.
                        idx (int): The index of the fish to respawn.
                        key_local (jax.random.PRNGKey): A JAX random key for generating random values.

                    Returns:
                        Tuple[chex.Array, chex.Array]: A tuple containing:
                            - The updated positions of all fish.
                            - The updated directions of all fish.
                    """
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

            # P2 Logic (mirror of P1, with some differences)
            def p2_auto_lower(_):
                """
                    Automatically lowers Player 2's hook until it reaches the water surface.

                    This function increments the vertical position of the hook by the auto-lower speed
                    and resets the hook's velocity. If the hook reaches the water surface, its state is
                    updated to indicate that it is no longer in the auto-lowering phase.

                    Args:
                        _ (Any): Placeholder argument, not used in the function.

                    Returns:
                        Tuple[float, float, int]: A tuple containing:
                            - The updated vertical position of the hook.
                            - The updated vertical velocity of the hook (always 0.0).
                            - The updated hook state (0 if the hook reaches the water surface, otherwise unchanged).
                    """
                new_y = p2.hook_y + cfg.AUTO_LOWER_SPEED
                new_vel_y = 0.0
                hook_reached_water = new_y >= water_surface_hook_y
                final_state = jnp.where(hook_reached_water, 0, p2.hook_state)
                final_y = jnp.where(hook_reached_water, water_surface_hook_y, new_y)
                return final_y, new_vel_y, final_state

            def p2_normal(_):
                """
                    Handles the normal vertical movement of Player 2's hook.

                    This function calculates the vertical velocity and position of the hook
                    based on Player 2's input actions. It ensures that the hook's movement
                    respects the acceleration, damping, and boundary constraints. The hook
                    can only move vertically when it is in the free state (hook_state == 0).

                    Args:
                        _ (Any): Placeholder argument, not used in the function.

                    Returns:
                        Tuple[float, float, int]: A tuple containing:
                            - The updated vertical position of the hook.
                            - The updated vertical velocity of the hook.
                            - The updated hook state (unchanged in this function).
                    """
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

            # P2 Collision and Game Logic
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
            # Updated P2-Hook-Position
            p2_hook_x, p2_hook_y = self._get_hook_position_p2(cfg.P2_START_X, PlayerState(
                rod_length=p2_new_rod_length, hook_y=p2_new_hook_y, score=p2.score, hook_state=p2_hook_state,
                hooked_fish_idx=p2_hooked_fish_idx, hook_velocity_y=p2_new_hook_velocity_y,
                hook_x_offset=p2_new_hook_x_offset, display_score=p2.display_score,
                score_animation_timer=p2.score_animation_timer, line_segments_x=p2.line_segments_x
            ))

            # If P2 has a fish: Fish follows / Boundaries / Wobble
            p2_has_hook = (p2_hook_state > 0) & (p2_hooked_fish_idx >= 0)

            def p2_update_hooked():
                """
                    Updates the position and behavior of a hooked fish for Player 2.

                    This function calculates the new position of the hooked fish based on its
                    current position, direction, and wobble effect. It also ensures the fish
                    stays within the boundaries defined by the rod's reach and the global game
                    boundaries.

                    Returns:
                        None
                    """
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
                    """
                            Applies boundaries specific to the hooked fish.

                            This function ensures the hooked fish stays within the boundaries
                            defined by the rod's reach. If the fish hits the boundary, its direction
                            is reversed, and an elastic push effect is applied.

                            Returns:
                                Tuple[float, float]: The constrained x-coordinate of the fish and its new direction.
                            """
                    would_hit_left = potential_new_x <= boundary_min
                    would_hit_right = potential_new_x >= boundary_max
                    constrained_x = jnp.clip(potential_new_x, boundary_min, boundary_max)
                    new_direction = jnp.where(would_hit_left | would_hit_right, -fish_dir, fish_dir)
                    elastic_push = jnp.where(would_hit_left, 0.5, jnp.where(would_hit_right, -0.5, 0.0))
                    return constrained_x + elastic_push, new_direction

                def apply_global_boundaries():
                    """
                            Applies global boundaries to the hooked fish.

                            This function ensures the hooked fish stays within the global game
                            boundaries. If the fish hits the boundary, its direction is reversed.

                            Returns:
                                Tuple[float, float]: The constrained x-coordinate of the fish and its new direction.
                            """
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

            # P2 Scoring / Shark-Kollision
            p2_score = p2.score
            p2_has_hooked_fish = (p2_hook_state > 0) & (p2_hooked_fish_idx >= 0)

            def p2_fpos():
                """
                    Retrieves the position of the fish currently hooked by Player 2.

                    This function returns the x and y coordinates of the fish that Player 2
                    has hooked. If no fish is hooked, the function defaults to returning (0.0, 0.0).

                    Returns:
                        Tuple[float, float]: A tuple containing:
                            - The x-coordinate of the hooked fish.
                            - The y-coordinate of the hooked fish.
                    """
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

            # Game Over first to 99 points
            game_over = (p1_score >= 99) | (p2_score >= 99)


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


class FishingDerbyRenderer(JAXGameRenderer):
    """
    Renders the Fishing Derby game state.

    Missing: water shimmer effect
    """

    def __init__(self, consts: Optional[GameConfig] = None):
        super().__init__()
        self.consts = consts or GameConfig.create_default()

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.sprite_path = os.path.join(module_dir, "sprites/fishingderby")

        final_asset_config = list(self.consts.ASSET_CONFIG)
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        self.LINE_P1_ID = self.COLOR_TO_ID.get((255, 255, 0), 0)
        self.LINE_P2_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)
        self.HOOK_ID = self.COLOR_TO_ID.get((255, 255, 255), 0)

        self.SPRITE_SHARK = self.SHAPE_MASKS['shark']
        self.SPRITE_FISH = self.SHAPE_MASKS['fish']

    def _get_hook_position(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        cfg = self.consts
        rod_end_x = player_x + player_state.rod_length
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    def _get_hook_position_p2(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        cfg = self.consts
        rod_end_x = player_x - player_state.rod_length
        hook_x = rod_end_x + player_state.hook_x_offset
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> chex.Array:
        """
        Renders the current game state into a raster image.

        WARNING: This is an incomplete render. See class docstring for details.
        """
        cfg = self.consts

        raster = self.jr.create_object_raster(self.BACKGROUND)

        raster = self.jr.render_at(
            raster,
            cfg.P1_START_X,
            cfg.PLAYER_Y,
            self.SHAPE_MASKS['player1'],
            flip_offset=self.FLIP_OFFSETS['player1'],
        )
        raster = self.jr.render_at(
            raster,
            cfg.P2_START_X,
            cfg.PLAYER_Y,
            self.SHAPE_MASKS['player2'],
            flip_offset=self.FLIP_OFFSETS['player2'],
        )

        shark_frame_idx = (state.time // 4) % self.SPRITE_SHARK.shape[0]
        shark_frame = self.SPRITE_SHARK[shark_frame_idx]
        raster = self.jr.render_at(
            raster,
            state.shark_x,
            cfg.SHARK_Y,
            shark_frame,
            flip_horizontal=state.shark_dir < 0,
            flip_offset=self.FLIP_OFFSETS['shark'],
        )

        fish_frame_idx = (state.time // 5) % self.SPRITE_FISH.shape[0]
        fish_frame = self.SPRITE_FISH[fish_frame_idx]
        hooked_fish_frame = self.SPRITE_FISH[(state.time // 6) % self.SPRITE_FISH.shape[0]]

        def draw_one_fish(i, r):
            pos = state.fish_positions[i]
            direction = state.fish_directions[i]
            active = state.fish_active[i]
            is_hooked_p1 = (state.p1.hooked_fish_idx == i) & (state.p1.hook_state > 0)
            is_hooked_p2 = (state.p2.hooked_fish_idx == i) & (state.p2.hook_state > 0)
            is_hooked = is_hooked_p1 | is_hooked_p2

            frame_to_use = jax.lax.cond(is_hooked, lambda: hooked_fish_frame, lambda: fish_frame)
            flip_sprite = direction > 0

            def render_active(raster_in):
                return self.jr.render_at(
                    raster_in,
                    pos[0],
                    pos[1],
                    frame_to_use,
                    flip_horizontal=flip_sprite,
                    flip_offset=self.FLIP_OFFSETS['fish'],
                )

            return jax.lax.cond(active, render_active, lambda raster_in: raster_in, r)

        raster = jax.lax.fori_loop(0, cfg.NUM_FISH, draw_one_fish, raster)

        def draw_hooked_p1(r):
            fish_idx = state.p1.hooked_fish_idx
            fish_pos = state.fish_positions[fish_idx]
            fish_dir = state.fish_directions[fish_idx]
            flip_sprite = fish_dir > 0
            frame = self.SPRITE_FISH[(state.time // 2) % self.SPRITE_FISH.shape[0]]
            return self.jr.render_at(
                r,
                fish_pos[0],
                fish_pos[1],
                frame,
                flip_horizontal=flip_sprite,
                flip_offset=self.FLIP_OFFSETS['fish'],
            )

        should_draw_hooked_p1 = (state.p1.hook_state > 0) & (state.p1.hooked_fish_idx >= 0) & (state.p1.hook_state != 3)
        raster = jax.lax.cond(should_draw_hooked_p1, draw_hooked_p1, lambda r: r, raster)

        def draw_hooked_p2(r):
            fish_idx = state.p2.hooked_fish_idx
            fish_pos = state.fish_positions[fish_idx]
            fish_dir = state.fish_directions[fish_idx]
            flip_sprite = fish_dir > 0
            frame = self.SPRITE_FISH[(state.time // 2) % self.SPRITE_FISH.shape[0]]
            return self.jr.render_at(
                r,
                fish_pos[0],
                fish_pos[1],
                frame,
                flip_horizontal=flip_sprite,
                flip_offset=self.FLIP_OFFSETS['fish'],
            )

        should_draw_hooked_p2 = (state.p2.hook_state > 0) & (state.p2.hooked_fish_idx >= 0) & (state.p2.hook_state != 3)
        raster = jax.lax.cond(should_draw_hooked_p2, draw_hooked_p2, lambda r: r, raster)

        raster = self.jr.render_at(
            raster,
            0,
            cfg.WATER_Y_START - 10,
            self.SHAPE_MASKS['pier'],
            flip_offset=self.FLIP_OFFSETS['pier'],
        )

        raster = self._render_score(raster, state.p1.display_score, 50, 10)
        raster = self._render_score(raster, state.p2.display_score, 100, 10)

        # Draw fishing lines (before palette conversion)
        raster = self._draw_fishing_lines(raster, state)

        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_score(self, raster, display_score, x, y):
        """Renders the player's score using the new API."""
        digit_masks = self.SHAPE_MASKS['score_digits']
        s1 = display_score // 10
        s0 = display_score % 10

        raster = self.jr.render_at(
            raster,
            x,
            y,
            digit_masks[s1],
            flip_offset=self.FLIP_OFFSETS['score_digits'],
        )
        raster = self.jr.render_at(
            raster,
            x + 7,
            y,
            digit_masks[s0],
            flip_offset=self.FLIP_OFFSETS['score_digits'],
        )
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _draw_fishing_lines(self, raster, state):
        """Draws fishing lines for both players on the palette-ID raster."""
        cfg = self.consts
        
        # Get scaling factors for downscaling
        width_scale = jnp.array(self.jr.config.width_scaling, dtype=jnp.float32)
        height_scale = jnp.array(self.jr.config.height_scaling, dtype=jnp.float32)

        # Player 1 fishing line
        p1_rod_end_x = cfg.P1_START_X + state.p1.rod_length
        hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, state.p1)

        # Scale coordinates for downscaling
        p1_rod_start_x_scaled = jnp.round((cfg.P1_START_X + 7) * width_scale).astype(jnp.int32)
        p1_rod_start_y_scaled = jnp.round(cfg.ROD_Y * height_scale).astype(jnp.int32)
        p1_rod_end_x_scaled = jnp.round(p1_rod_end_x * width_scale).astype(jnp.int32)
        p1_rod_end_y_scaled = jnp.round(cfg.ROD_Y * height_scale).astype(jnp.int32)
        hook_x_scaled = jnp.round(hook_x * width_scale).astype(jnp.int32)
        hook_y_scaled = jnp.round(hook_y * height_scale).astype(jnp.int32)

        # Draw horizontal part of Player 1 rod (black)
        raster = self._render_line_id(raster, p1_rod_start_x_scaled, p1_rod_start_y_scaled, 
                                      p1_rod_end_x_scaled, p1_rod_end_y_scaled, self.LINE_P2_ID)

        # Player 1 line rendering logic
        in_water = state.p1.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
        is_reeling = state.p1.hook_state > 0
        has_horizontal_offset = jnp.abs(state.p1.hook_x_offset) > 0.5
        apply_sag = in_water & ~is_reeling & has_horizontal_offset

        water_depth_ratio = jnp.clip((state.p1.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0, 1.0)
        # Scale sag amount proportionally (use average of width and height scaling for sag)
        sag_scale = (width_scale + height_scale) / 2.0
        sag_amount = jnp.where(apply_sag, (3.0 + water_depth_ratio * 3.0) * sag_scale, 0.0)

        line_start = jnp.array([p1_rod_end_x_scaled, p1_rod_end_y_scaled], dtype=jnp.float32)
        line_end = jnp.array([hook_x_scaled, hook_y_scaled], dtype=jnp.float32)

        raster = jax.lax.cond(
            apply_sag,
            lambda r: self._render_saggy_line_id(r, line_start, line_end, sag_amount, self.LINE_P1_ID, sag_scale),
            lambda r: self._render_line_id(r, line_start[0], line_start[1], line_end[0], line_end[1], self.LINE_P1_ID),
            raster
        )

        # Player 2 fishing line
        p2_rod_end_x = cfg.P2_START_X - state.p2.rod_length
        p2_hook_x, p2_hook_y = self._get_hook_position_p2(cfg.P2_START_X, state.p2)

        # Scale coordinates for downscaling
        p2_rod_start_x_scaled = jnp.round((cfg.P2_START_X + 8) * width_scale).astype(jnp.int32)
        p2_rod_start_y_scaled = jnp.round(cfg.ROD_Y * height_scale).astype(jnp.int32)
        p2_rod_end_x_scaled = jnp.round(p2_rod_end_x * width_scale).astype(jnp.int32)
        p2_rod_end_y_scaled = jnp.round(cfg.ROD_Y * height_scale).astype(jnp.int32)
        p2_hook_x_scaled = jnp.round(p2_hook_x * width_scale).astype(jnp.int32)
        p2_hook_y_scaled = jnp.round(p2_hook_y * height_scale).astype(jnp.int32)

        # Draw horizontal part of Player 2 rod (black)
        raster = self._render_line_id(raster, p2_rod_start_x_scaled, p2_rod_start_y_scaled,
                                      p2_rod_end_x_scaled, p2_rod_end_y_scaled, self.LINE_P2_ID)

        # Player 2 line rendering logic
        p2_in_water = state.p2.hook_y > (cfg.WATER_Y_START - cfg.ROD_Y)
        p2_is_reeling = state.p2.hook_state > 0
        p2_has_horizontal_offset = jnp.abs(state.p2.hook_x_offset) > 0.5
        p2_apply_sag = p2_in_water & ~p2_is_reeling & p2_has_horizontal_offset
        p2_water_depth_ratio = jnp.clip((state.p2.hook_y - (cfg.WATER_Y_START - cfg.ROD_Y)) / cfg.MAX_HOOK_DEPTH_Y, 0.0, 1.0)
        p2_sag_amount = jnp.where(p2_apply_sag, (3.0 + p2_water_depth_ratio * 3.0) * sag_scale, 0.0)
        p2_line_start = jnp.array([p2_rod_end_x_scaled, p2_rod_end_y_scaled], dtype=jnp.float32)
        p2_line_end = jnp.array([p2_hook_x_scaled, p2_hook_y_scaled], dtype=jnp.float32)

        raster = jax.lax.cond(
            p2_apply_sag,
            lambda r: self._render_saggy_line_id(r, p2_line_start, p2_line_end, p2_sag_amount, self.LINE_P2_ID, sag_scale),
            lambda r: self._render_line_id(r, p2_line_start[0], p2_line_start[1], p2_line_end[0], p2_line_end[1], self.LINE_P2_ID),
            raster
        )

        return raster

    @staticmethod
    @jax.jit
    def _render_line_id(raster, x0, y0, x1, y1, color_id):
        """Renders a straight line on a palette-ID raster using Bresenham's algorithm."""
        x0, y0, x1, y1 = jnp.round(jnp.array([x0, y0, x1, y1])).astype(jnp.int32)
        dx, sx, dy, sy = jnp.abs(x1 - x0), jnp.sign(x1 - x0), -jnp.abs(y1 - y0), jnp.sign(y1 - y0)
        err = dx + dy
        color_id_val = jnp.asarray(color_id, dtype=raster.dtype)

        def loop_body(carry):
            x, y, r, e = carry
            safe_y, safe_x = jnp.clip(y, 0, r.shape[0] - 1), jnp.clip(x, 0, r.shape[1] - 1)
            r = r.at[safe_y, safe_x].set(color_id_val)
            e2 = 2 * e
            e_new = jnp.where(e2 >= dy, e + dy, e)
            x_new = jnp.where(e2 >= dy, x + sx, x)
            e_final = jnp.where(e2 <= dx, e_new + dx, e_new)
            y_new = jnp.where(e2 <= dx, y + sy, y)
            return x_new, y_new, r, e_final

        def loop_cond(carry):
            return ~((carry[0] == x1) & (carry[1] == y1))

        _, _, raster, _ = jax.lax.while_loop(loop_cond, loop_body, (x0, y0, raster, err))

        safe_y1, safe_x1 = jnp.clip(y1, 0, raster.shape[0] - 1), jnp.clip(x1, 0, raster.shape[1] - 1)
        raster = raster.at[safe_y1, safe_x1].set(color_id_val)
        return raster

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def _render_saggy_line_id(raster, p_start, p_end, sag_amount, color_id, scale_factor, num_segments=10):
        """Renders a fishing line with sagging effect on a palette-ID raster."""
        # Create 't' values from 0.0 to 1.0 to parameterize the curve
        t = jnp.linspace(0.0, 1.0, num_segments + 1)

        # Linearly interpolate between start and end points
        points = jax.vmap(lambda i: p_start + i * (p_end - p_start))(t)

        # Calculate line properties
        line_vector = p_end - p_start
        line_length = jnp.linalg.norm(line_vector) + 1e-8

        # Sag calculation - scale the length threshold to match the coordinate scaling
        scaled_length_threshold = 50.0 * scale_factor
        parabolic_sag = 4.0 * t * (1.0 - t)
        catenary_factor = jnp.cosh(3.0 * (t - 0.5)) - 1.0
        normalized_catenary = catenary_factor / (jnp.max(catenary_factor) + 1e-8)
        length_factor = jnp.clip(line_length / scaled_length_threshold, 0.5, 2.0)
        total_sag = sag_amount * (0.8 * parabolic_sag + 0.2 * normalized_catenary) / length_factor

        # Sag direction
        is_nearly_vertical = jnp.abs(line_vector[0]) < 0.1
        line_direction_norm = line_vector / line_length
        perpendicular = jnp.array([-line_direction_norm[1], line_direction_norm[0]])

        sag_offsets_x = jnp.where(is_nearly_vertical, total_sag, total_sag * perpendicular[0])
        sag_offsets_y = jnp.where(is_nearly_vertical, jnp.zeros_like(total_sag), total_sag * perpendicular[1])

        # Apply sag offsets
        points = points.at[:, 0].add(sag_offsets_x)
        points = points.at[:, 1].add(sag_offsets_y)

        # Draw segments
        def draw_segment(i, current_raster):
            p1 = points[i]
            p2 = points[i + 1]
            return FishingDerbyRenderer._render_line_id(
                current_raster, p1[0], p1[1], p2[0], p2[1], color_id
            )

        raster = jax.lax.fori_loop(0, num_segments, draw_segment, raster)
        return raster