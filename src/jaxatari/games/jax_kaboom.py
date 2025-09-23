from functools import partial
from typing import NamedTuple, Tuple, Dict
import os
import jax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr

WIDTH = 160
HEIGHT = 210
MAX_BOMBS = 8
MAX_STEPS = 10000

class KaboomConstants(NamedTuple):
    WIDTH: int = WIDTH
    HEIGHT: int = HEIGHT
    MAX_BOMBS: int = MAX_BOMBS
    MAX_STEPS: int = MAX_STEPS

    BUCKET_WIDTH: int = 8
    BUCKET_HEIGHT: int = 6
    BUCKET_SPEED: int = 3

BUCKET_START_X: int = WIDTH // 2
BUCKET_START_Y: int = HEIGHT - 30

# Bucket dimensions and speed (module-level for use across functions)
BUCKET_WIDTH: int = 8
BUCKET_HEIGHT: int = 6
BUCKET_SPEED: int = 3

BOMBER_WIDTH: int = 8
BOMBER_HEIGHT: int = 8
BOMBER_START_X: int = WIDTH // 2
BOMBER_START_Y: int = 30
BOMBER_MIN_X: int = 20
BOMBER_MAX_X: int = WIDTH - 20
BOMBER_SPEED: int = 2

BOMB_WIDTH: int = 4
BOMB_HEIGHT: int = 4
BOMB_SPEED_INITIAL: int = 1
BOMB_SPEED_MAX: int = 4
BOMB_SPEED_INCREMENT: float = 0.1

INITIAL_LIVES: int = 3
POINTS_PER_CATCH: int = 10
MAX_SCORE: int = 999999  # Maximum score as per original Atari Kaboom

DIFFICULTY_NORMAL: int = 0  # Full-size buckets (b position)
DIFFICULTY_ADVANCED: int = 1  # Half-size buckets (a position)

BOMB_GROUPS = [
    (1, 10, 1, 10, 10),
    (2, 20, 2, 40, 50),
    (3, 30, 3, 90, 140),
    (4, 40, 4, 160, 300),
    (5, 50, 5, 250, 550),
    (6, 75, 6, 450, 1000),
    (7, 100, 7, 700, 1700),
    (8, 150, 8, 1200, 2900)
]

GROUP_BOMB_COUNTS = jnp.array([group[1] for group in BOMB_GROUPS])
GROUP_POINT_VALUES = jnp.array([group[2] for group in BOMB_GROUPS])
GROUP_CUMULATIVE_SCORES = jnp.array([group[4] for group in BOMB_GROUPS])

GROUP_SPEED_MULTIPLIERS = jnp.array([1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.0])

# Colors
BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
BUCKET_COLOR: Tuple[int, int, int] = (255, 255, 0)
BOMBER_COLOR: Tuple[int, int, int] = (255, 0, 0)
BOMB_COLOR: Tuple[int, int, int] = (255, 165, 0)
TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
GROUP_TRANSITION_PAUSE = 120


STATE_TRANSLATOR: Dict[int, str] = {
    0: "bucket_x",
    1: "bomber_x",
    2: "bomber_direction",
    3: "score",
    4: "lives",
    5: "level",
    6: "step_counter",
}

for i in range(MAX_BOMBS):
    base_idx = 7 + i * 4
    STATE_TRANSLATOR[base_idx] = f"bomb_{i}_active"
    STATE_TRANSLATOR[base_idx + 1] = f"bomb_{i}_x"
    STATE_TRANSLATOR[base_idx + 2] = f"bomb_{i}_y"
    STATE_TRANSLATOR[base_idx + 3] = f"bomb_{i}_speed"

# Calculate cumulative bombs to reach each group
BOMB_GROUP_THRESHOLDS = jnp.array([0] + [sum(GROUP_BOMB_COUNTS[:i]) for i in range(1, len(GROUP_BOMB_COUNTS) + 1)])

# Load digit sprites once for JIT-safe score rendering
MODULE_DIR = os.path.dirname(__file__)
# Reuse an existing digit set (Seaquest) to avoid adding new assets
_DIGITS_PATH = os.path.join(MODULE_DIR, "sprites", "seaquest", "digits", "{}.npy")
try:
    DIGIT_SPRITES = jr.load_and_pad_digits(_DIGITS_PATH, num_chars=10)
except Exception:
    # Fallback to a minimal 1x1 white pixel per digit to avoid crashes if files are missing
    DIGIT_SPRITES = jnp.ones((10, 1, 1, 4), dtype=jnp.uint8) * jnp.array([255, 255, 255, 255], dtype=jnp.uint8)

# Immutable state container
class KaboomState(NamedTuple):
    bucket_x: chex.Array
    bomber_x: chex.Array
    bomber_direction: chex.Array
    score: chex.Array
    lives: chex.Array
    level: chex.Array
    step_counter: chex.Array
    bomb_active: chex.Array
    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_speed: chex.Array
    drop_timer: chex.Array
    obs_stack: chex.ArrayTree
    bombs_caught: chex.Array
    bomb_group: chex.Array
    bombs_in_group: chex.Array
    difficulty: chex.Array
    group_transition_timer: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class BombsObservation(NamedTuple):
    active: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class KaboomObservation(NamedTuple):
    bucket: EntityPosition
    bomber: EntityPosition
    bombs: BombsObservation
    score: jnp.ndarray
    lives: jnp.ndarray
    level: jnp.ndarray

class KaboomInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array
    lives: jnp.ndarray
    score: jnp.ndarray
    level: jnp.ndarray
    bombs_caught: jnp.ndarray
    bomb_group: jnp.ndarray
    bombs_in_group: jnp.ndarray

@jax.jit
def bucket_step(bucket_x: chex.Array, action: chex.Array) -> chex.Array:
    """Update bucket position based on action."""
    bucket_x = jnp.where(
        jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE),
        bucket_x - BUCKET_SPEED,
        bucket_x
    )

    bucket_x = jnp.where(
        jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE),
        bucket_x + BUCKET_SPEED,
        bucket_x
    )

    bucket_x = jnp.clip(bucket_x, 0, WIDTH - BUCKET_WIDTH)
    return bucket_x

@jax.jit
def bomber_step(key: chex.PRNGKey, bomber_x: chex.Array, bomber_direction: chex.Array) -> Tuple[chex.PRNGKey, chex.Array, chex.Array]:
    """Update bomber position with random movement."""
    key, subkey = jax.random.split(key)
    change_direction_prob = 0.1
    random_value = jax.random.uniform(subkey, shape=())
    should_change_direction = random_value < change_direction_prob

    new_bomber_direction = jnp.where(
        should_change_direction,
        -bomber_direction,
        bomber_direction
    )

    new_bomber_x = bomber_x + (new_bomber_direction * BOMBER_SPEED)

    hit_left_edge = new_bomber_x <= BOMBER_MIN_X
    hit_right_edge = new_bomber_x >= BOMBER_MAX_X
    hit_edge = jnp.logical_or(hit_left_edge, hit_right_edge)

    new_bomber_direction = jnp.where(
        hit_edge,
        -new_bomber_direction,
        new_bomber_direction
    )

    new_bomber_x = jnp.clip(new_bomber_x, BOMBER_MIN_X, BOMBER_MAX_X)
    return key, new_bomber_x, new_bomber_direction

@jax.jit
def drop_bomb(
    bomb_active: chex.Array,
    bomb_x: chex.Array,
    bomb_y: chex.Array,
    bomb_speed: chex.Array,
    bomber_x: chex.Array,
    level: chex.Array,
    drop_timer: chex.Array,
    bomb_group: chex.Array,
    pause_dropping: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Potentially drop a new bomb."""
    # Skip decrement/drop during transition pause
    new_drop_timer = jnp.where(pause_dropping > 0, drop_timer, drop_timer - 1)

    should_drop = jnp.logical_and(new_drop_timer <= 0, pause_dropping <= 0)

    inactive_slots = jnp.where(bomb_active == 0, 1, 0)
    first_inactive = jnp.argmax(inactive_slots)

    can_drop = jnp.sum(bomb_active) < MAX_BOMBS
    will_drop = jnp.logical_and(should_drop, can_drop)

    # Group-based speed and drop interval
    safe_group_idx = jnp.clip(bomb_group, 0, len(GROUP_SPEED_MULTIPLIERS) - 1)
    current_speed_multiplier = jnp.where(
        bomb_group < len(GROUP_SPEED_MULTIPLIERS),
        GROUP_SPEED_MULTIPLIERS[safe_group_idx],
        GROUP_SPEED_MULTIPLIERS[-1]
    )

    current_drop_interval = jnp.maximum(18 - bomb_group, 5)

    new_drop_timer = jnp.where(
        will_drop,
        current_drop_interval,
        new_drop_timer
    )

    base_speed = BOMB_SPEED_INITIAL * current_speed_multiplier
    adjusted_speed = jnp.minimum(base_speed + level * 0.03, BOMB_SPEED_MAX)

    new_bomb_active = bomb_active.at[first_inactive].set(
        jnp.where(will_drop, 1, bomb_active[first_inactive])
    )
    new_bomb_x = bomb_x.at[first_inactive].set(
        jnp.where(will_drop, bomber_x, bomb_x[first_inactive])
    )
    new_bomb_y = bomb_y.at[first_inactive].set(
        jnp.where(will_drop, BOMBER_START_Y + BOMBER_HEIGHT, bomb_y[first_inactive])
    )
    new_bomb_speed = bomb_speed.at[first_inactive].set(
        jnp.where(will_drop, adjusted_speed, bomb_speed[first_inactive])
    )
    return new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed, new_drop_timer

@jax.jit
def update_bombs(
    bomb_active: chex.Array,
    bomb_x: chex.Array,
    bomb_y: chex.Array,
    bomb_speed: chex.Array,
    bucket_x: chex.Array,
    score: chex.Array,
    lives: chex.Array,
    level: chex.Array,
    bombs_caught: chex.Array,
    bomb_group: chex.Array,
    bombs_in_group: chex.Array,
    difficulty: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Update positions of active bombs and check for catches/misses."""

    new_bomb_y = jnp.where(
        bomb_active == 1,
        bomb_y + bomb_speed,
        bomb_y
    )

    effective_bucket_width = jnp.where(
        difficulty == DIFFICULTY_ADVANCED,
        BUCKET_WIDTH // 2,
        BUCKET_WIDTH
    )

    bucket_left = bucket_x
    bucket_right = bucket_x + effective_bucket_width

    caught = jnp.zeros_like(bomb_active, dtype=jnp.bool_)

    def check_bucket_collision(bucket_index, caught_bombs):
        if bucket_index != 0:
            return caught_bombs

        bucket_visible = bucket_index < lives

        bucket_top = BUCKET_START_Y
        bucket_bottom = bucket_top + BUCKET_HEIGHT

        bomb_in_bucket_x = jnp.logical_and(
            bomb_x + BOMB_WIDTH >= bucket_left - 5,
            bomb_x <= bucket_right + 5
        )
        bomb_in_bucket_y = jnp.logical_and(
            new_bomb_y + BOMB_HEIGHT >= bucket_top - 3,
            new_bomb_y <= bucket_bottom + 3
        )
        bucket_collision = jnp.logical_and(
            bomb_active == 1,
            jnp.logical_and(bomb_in_bucket_x, bomb_in_bucket_y)
        )

        valid_catches = jnp.logical_and(bucket_collision, bucket_visible)
        return jnp.logical_or(caught_bombs, valid_catches)

    caught = check_bucket_collision(0, caught)

    missed = jnp.logical_and(
        bomb_active == 1,
        new_bomb_y > HEIGHT
    )

    any_bomb_missed = jnp.any(missed)

    new_bomb_active = jnp.where(
        any_bomb_missed,
        jnp.zeros_like(bomb_active),
        jnp.where(caught, 0, bomb_active)
    )

    new_lives = jnp.where(any_bomb_missed, lives - 1, lives)

    catches = jnp.sum(caught)

    safe_group_idx = jnp.clip(bomb_group, 0, len(GROUP_POINT_VALUES) - 1)

    current_points_per_bomb = jnp.where(
        bomb_group < len(GROUP_POINT_VALUES),
        GROUP_POINT_VALUES[safe_group_idx],
        GROUP_POINT_VALUES[-1]
    )

    points_earned = catches * current_points_per_bomb

    new_score = jnp.minimum(score + points_earned, MAX_SCORE)

    new_bombs_caught = bombs_caught + catches

    new_bombs_in_group = bombs_in_group + catches

    current_group_bomb_limit = jnp.where(
        bomb_group < len(GROUP_BOMB_COUNTS),
        GROUP_BOMB_COUNTS[safe_group_idx],
        jnp.inf
    )
    advance_group = new_bombs_in_group >= current_group_bomb_limit

    new_bomb_group = jnp.where(
        advance_group,
        bomb_group + 1,
        bomb_group
    )

    new_bombs_in_group = jnp.where(
        advance_group,
        0,
        new_bombs_in_group
    )

    new_group_transition_timer = jnp.where(
        advance_group,
        GROUP_TRANSITION_PAUSE,
        0
    )

    new_level = (jnp.floor(new_score / 100) + 1).astype(jnp.int32)
    
    return (
        new_bomb_active, 
        bomb_x, 
        new_bomb_y, 
        bomb_speed, 
        new_score, 
        new_lives, 
        new_level, 
        new_bombs_caught, 
        new_bomb_group, 
        new_bombs_in_group,
        new_group_transition_timer
    )

@jax.jit
def step_fn(key: chex.PRNGKey, state: KaboomState, action: chex.Array) -> Tuple[chex.PRNGKey, KaboomState, chex.Array, chex.Array, KaboomInfo]:
    """Advance the game state by one step."""
    key, bomber_key = jax.random.split(key)
    new_bucket_x = bucket_step(state.bucket_x, action)
    bomber_key, new_bomber_x, new_bomber_direction = bomber_step(bomber_key, state.bomber_x, state.bomber_direction)

    new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed, new_drop_timer = drop_bomb(
        state.bomb_active, state.bomb_x, state.bomb_y, state.bomb_speed,
        new_bomber_x, state.level, state.drop_timer, state.bomb_group,
        state.group_transition_timer
    )

    new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed, new_score, new_lives, new_level, new_bombs_caught, new_bomb_group, new_bombs_in_group, new_group_transition_timer = update_bombs(
        new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed,
        new_bucket_x, state.score, state.lives, state.level,
        state.bombs_caught, state.bomb_group, state.bombs_in_group, state.difficulty
    )

    reward = new_score - state.score

    done = new_lives <= 0

    # Create updated state and decrement transition timer if active
    dec_group_transition_timer = jnp.maximum(state.group_transition_timer - 1, 0)

    new_state = KaboomState(
        bucket_x=new_bucket_x,
        bomber_x=new_bomber_x,
        bomber_direction=new_bomber_direction,
        score=new_score,
        lives=new_lives,
        level=new_level,
        step_counter=state.step_counter + 1,
        bomb_active=new_bomb_active,
        bomb_x=new_bomb_x,
        bomb_y=new_bomb_y,
        bomb_speed=new_bomb_speed,
        drop_timer=new_drop_timer,
        obs_stack=state.obs_stack,
        bombs_caught=new_bombs_caught,
        bomb_group=new_bomb_group,
        bombs_in_group=new_bombs_in_group,
        difficulty=state.difficulty,
        group_transition_timer=jnp.maximum(new_group_transition_timer, dec_group_transition_timer)
    )

    info = KaboomInfo(
        time=state.step_counter,
        all_rewards=reward,
        lives=new_lives,
        score=new_score,
        level=new_level,
        bombs_caught=new_bombs_caught,
        bomb_group=new_bomb_group,
        bombs_in_group=new_bombs_in_group
    )
    
    return key, new_state, reward, done, info

@jax.jit
def reset_fn(key: chex.PRNGKey) -> KaboomState:
    """Initialize a new game state."""
    key, subkey = jax.random.split(key)

    bomber_direction = jax.random.choice(subkey, jnp.array([-1, 1]))

    bomb_active = jnp.zeros(MAX_BOMBS, dtype=jnp.int32)
    bomb_x = jnp.zeros(MAX_BOMBS, dtype=jnp.float32)
    bomb_y = jnp.zeros(MAX_BOMBS, dtype=jnp.float32)
    bomb_speed = jnp.ones(MAX_BOMBS, dtype=jnp.float32) * BOMB_SPEED_INITIAL

    state = KaboomState(
        bucket_x=jnp.array(BUCKET_START_X, dtype=jnp.float32),
        bomber_x=jnp.array(BOMBER_START_X, dtype=jnp.float32),
        bomber_direction=bomber_direction,
        score=jnp.array(0, dtype=jnp.int32),
        lives=jnp.array(INITIAL_LIVES, dtype=jnp.int32),
        level=jnp.array(1, dtype=jnp.int32),
        step_counter=jnp.array(0, dtype=jnp.int32),
        bomb_active=bomb_active,
        bomb_x=bomb_x,
        bomb_y=bomb_y,
        bomb_speed=bomb_speed,
        drop_timer=jnp.array(12, dtype=jnp.int32),
        obs_stack=None,
        bombs_caught=jnp.array(0, dtype=jnp.int32),
        bomb_group=jnp.array(0, dtype=jnp.int32),
        bombs_in_group=jnp.array(0, dtype=jnp.int32),
        difficulty=jnp.array(DIFFICULTY_NORMAL, dtype=jnp.int32),
        group_transition_timer=jnp.array(0, dtype=jnp.int32)
    )
    
    return state

class JaxKaboom(JaxEnvironment[KaboomState, KaboomObservation, KaboomInfo, KaboomConstants]):
    """JAX implementation of the Kaboom game environment."""
    
    def __init__(self, consts: KaboomConstants = None):
        consts = consts or KaboomConstants()
        super().__init__(consts)
    
    @property
    def default_state(self) -> KaboomState:
        """Get the default initial state."""
        return reset_fn(jax.random.PRNGKey(0))
    
    def step_env(
        self, key: chex.PRNGKey, state: KaboomState, action: chex.Array
    ) -> Tuple[chex.PRNGKey, KaboomState, chex.Array, chex.Array, KaboomInfo]:
        """Take a step in the environment."""
        key, next_state, reward, done, info = step_fn(key, state, action)
        return key, next_state, reward, done, info
    
    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, KaboomState]:
        """Reset the environment."""
        key, subkey = jax.random.split(key)
        next_state = reset_fn(subkey)
        return key, next_state

    def reset(self, key: chex.PRNGKey = None) -> Tuple[KaboomObservation, KaboomState]:
        """Reset the environment to initial state."""
        if key is None:
            key = jax.random.PRNGKey(0)
        reset_key, state = self.reset_env(key)
        obs = self._get_observation(state)
        return obs, state

    def step(self, state: KaboomState, action: chex.Array) -> Tuple[KaboomObservation, KaboomState, float, bool, KaboomInfo]:
        """Take a step in the environment."""
        step_key = jax.random.PRNGKey(state.step_counter)
        step_key, next_state, reward, done, info = self.step_env(step_key, state, action)
        
        # Calculate all rewards for multi-reward logging
        all_rewards = self._get_all_reward(state, next_state)
        
        # Update info with proper all_rewards
        updated_info = KaboomInfo(
            time=info.time,
            all_rewards=all_rewards,
            lives=info.lives,
            score=info.score,
            level=info.level,
            bombs_caught=info.bombs_caught,
            bomb_group=info.bomb_group,
            bombs_in_group=info.bombs_in_group,
        )
        
        obs = self._get_observation(next_state)
        return obs, next_state, reward, done, updated_info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: KaboomState) -> KaboomObservation:
        """Convert state to structured observation."""
        effective_bucket_width = jnp.where(
            state.difficulty == DIFFICULTY_ADVANCED,
            BUCKET_WIDTH // 2,
            BUCKET_WIDTH
        ).astype(jnp.int32)

        bucket = EntityPosition(
            x=state.bucket_x.astype(jnp.int32),
            y=jnp.array(BUCKET_START_Y, dtype=jnp.int32),
            width=effective_bucket_width,
            height=jnp.array(BUCKET_HEIGHT, dtype=jnp.int32)
        )

        bomber = EntityPosition(
            x=state.bomber_x.astype(jnp.int32),
            y=jnp.array(BOMBER_START_Y, dtype=jnp.int32),
            width=jnp.array(BOMBER_WIDTH, dtype=jnp.int32),
            height=jnp.array(BOMBER_HEIGHT, dtype=jnp.int32)
        )

        bombs = BombsObservation(
            active=state.bomb_active.astype(jnp.int32),
            x=state.bomb_x.astype(jnp.int32),
            y=state.bomb_y.astype(jnp.int32),
            width=jnp.ones((MAX_BOMBS,), dtype=jnp.int32) * BOMB_WIDTH,
            height=jnp.ones((MAX_BOMBS,), dtype=jnp.int32) * BOMB_HEIGHT,
        )

        return KaboomObservation(
            bucket=bucket,
            bomber=bomber,
            bombs=bombs,
            score=state.score.astype(jnp.int32),
            lives=state.lives.astype(jnp.int32),
            level=state.level.astype(jnp.int32)
        )
    
    @property
    def num_actions(self) -> int:
        """Number of possible actions."""
        return 6  # NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    
    def action_space(self) -> spaces.Discrete:
        """Action space definition."""
        return spaces.Discrete(self.num_actions)
    
    def observation_space(self) -> spaces.Dict:
        """Structured observation space definition."""
        return spaces.Dict({
            "bucket": spaces.Dict({
                "x": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "bomber": spaces.Dict({
                "x": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "bombs": spaces.Dict({
                "active": spaces.Box(low=0, high=1, shape=(MAX_BOMBS,), dtype=jnp.int32),
                "x": spaces.Box(low=0, high=WIDTH, shape=(MAX_BOMBS,), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=HEIGHT, shape=(MAX_BOMBS,), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=WIDTH, shape=(MAX_BOMBS,), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=HEIGHT, shape=(MAX_BOMBS,), dtype=jnp.int32),
            }),
            "score": spaces.Box(low=0, high=MAX_SCORE, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=INITIAL_LIVES, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=0, high=10000, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Image space definition."""
        return spaces.Box(0, 255, (HEIGHT, WIDTH, 3), dtype=jnp.uint8)
    
    def is_terminal(self, state: KaboomState) -> chex.Array:
        """Check if state is terminal."""
        return jnp.logical_or(
            state.lives <= 0,
            state.step_counter >= MAX_STEPS
        )

    def render(self, state: KaboomState) -> jnp.ndarray:
        """Render the current state as an RGB array."""
        return render_frame(state)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: KaboomObservation) -> jnp.ndarray:
        """Flatten structured observation into a 1D array."""
        return jnp.concatenate([
            obs.bucket.x.flatten(), obs.bucket.y.flatten(), obs.bucket.width.flatten(), obs.bucket.height.flatten(),
            obs.bomber.x.flatten(), obs.bomber.y.flatten(), obs.bomber.width.flatten(), obs.bomber.height.flatten(),
            obs.bombs.active.flatten(), obs.bombs.x.flatten(), obs.bombs.y.flatten(), obs.bombs.width.flatten(), obs.bombs.height.flatten(),
            obs.score.flatten(), obs.lives.flatten(), obs.level.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: KaboomState, all_rewards: chex.Array = None) -> KaboomInfo:
        return KaboomInfo(
            time=state.step_counter,
            all_rewards=jnp.array([0.0]) if all_rewards is None else all_rewards,
            lives=state.lives,
            score=state.score,
            level=state.level,
            bombs_caught=state.bombs_caught,
            bomb_group=state.bomb_group,
            bombs_in_group=state.bombs_in_group,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KaboomState, state: KaboomState) -> float:
        return (state.score - previous_state.score).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: KaboomState, state: KaboomState) -> jnp.ndarray:
        """Return array of rewards for multi-reward logging."""
        # Kaboom only has a single reward (score delta), so return a 1-element array
        base_reward = (state.score - previous_state.score).astype(jnp.float32)
        return jnp.array([base_reward])

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: KaboomState) -> bool:
        return jnp.logical_or(state.lives <= 0, state.step_counter >= MAX_STEPS)

@jax.jit
def render_frame(state: KaboomState) -> jnp.ndarray:
    """Render the current state as an RGB array (JIT-compatible)."""
    frame = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

    bucket_x = state.bucket_x.astype(jnp.int32)
    bucket_y = BUCKET_START_Y

    bucket_rect = jnp.ones((BUCKET_HEIGHT, BUCKET_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BUCKET_COLOR, dtype=jnp.uint8)

    frame = jax.lax.dynamic_update_slice(frame, bucket_rect, (bucket_y, bucket_x, 0))

    bomber_x = state.bomber_x.astype(jnp.int32)
    bomber_y = BOMBER_START_Y

    bomber_rect = jnp.ones((BOMBER_HEIGHT, BOMBER_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BOMBER_COLOR, dtype=jnp.uint8)

    frame = jax.lax.dynamic_update_slice(frame, bomber_rect, (bomber_y, bomber_x, 0))

    def draw_bomb(i, f):
        bomb_x = state.bomb_x[i].astype(jnp.int32)
        bomb_y = state.bomb_y[i].astype(jnp.int32)
        is_active = state.bomb_active[i] == 1

        use_bomb1 = (bomb_y // 8) % 2 == 1

        bomb_sprite_1 = jnp.ones((BOMB_HEIGHT, BOMB_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BOMB_COLOR, dtype=jnp.uint8)
        bomb_sprite_0 = jnp.ones((BOMB_HEIGHT, BOMB_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BOMB_COLOR, dtype=jnp.uint8)
        
        bomb_sprite = jax.lax.cond(
            use_bomb1,
            lambda: bomb_sprite_1,
            lambda: bomb_sprite_0
        )

        valid_coords = (
            is_active &
            (bomb_x >= 0) &
            (bomb_x + BOMB_WIDTH <= WIDTH) &
            (bomb_y >= 0) &
            (bomb_y + BOMB_HEIGHT <= HEIGHT)
        )

        f = jax.lax.cond(
            valid_coords,
            lambda _: jax.lax.dynamic_update_slice(f, bomb_sprite, (bomb_y, bomb_x, 0)),
            lambda _: f,
            operand=None
        )
        
        return f

    frame = jax.lax.fori_loop(0, MAX_BOMBS, draw_bomb, frame)

    # Top band (keep dark for score contrast)
    score_band = jnp.ones((8, WIDTH, 3), dtype=jnp.uint8) * jnp.array(BACKGROUND_COLOR, dtype=jnp.uint8)
    frame = jax.lax.dynamic_update_slice(frame, score_band, (0, 0, 0))

    # Render numeric score using shared digit sprites
    score_digits = jr.int_to_digits(state.score, max_digits=6)
    frame = jr.render_label(frame, 4, 2, score_digits, DIGIT_SPRITES, spacing=8)

    LIFE_BUCKET_HEIGHT = BUCKET_HEIGHT
    LIFE_BUCKET_WIDTH = BUCKET_WIDTH
    LIFE_SPACING = 5
    
    def draw_extra_life(i, f):
        life_number = i + 2
        has_life = life_number <= state.lives
        life_x = state.bucket_x.astype(jnp.int32)
        spacing = 5
        life_y = BUCKET_START_Y + BUCKET_HEIGHT + spacing + (i * (LIFE_BUCKET_HEIGHT + spacing))

        life_rect = jnp.ones((LIFE_BUCKET_HEIGHT, LIFE_BUCKET_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BUCKET_COLOR, dtype=jnp.uint8)
        f = jax.lax.cond(
            has_life,
            lambda _: jax.lax.dynamic_update_slice(f, life_rect, (life_y, life_x, 0)),
            lambda _: f,
            operand=None
        )
        
        return f

    frame = jax.lax.fori_loop(0, INITIAL_LIVES - 1, draw_extra_life, frame)
    
    return frame

def display_game(screen, state, fps_clock=None):
    """Display the game state on the screen (non-JIT function)."""
    import pygame
    import numpy as np
    
    # Render frame using the jitted function
    frame = render_frame(state)
    frame = np.array(frame)  # Convert to numpy for pygame
    
    # Transpose the frame to match pygame's expected dimensions (WIDTH, HEIGHT, 3)
    frame = np.transpose(frame, (1, 0, 2))
    
    # Scale the frame to screen size
    scale = 3
    surface = pygame.Surface((WIDTH, HEIGHT))
    pygame.surfarray.blit_array(surface, frame)
    scaled_surface = pygame.transform.scale(surface, (WIDTH * scale, HEIGHT * scale))
    screen.blit(scaled_surface, (0, 0))
    
    # Optional overlay (non-JIT): keep non-score info only, as score is now drawn in the JIT frame
    font = pygame.font.SysFont(None, 24)
    lives_text = font.render(f"Lives: {int(state.lives)}", True, TEXT_COLOR)
    level_text = font.render(f"Level: {int(state.level)}", True, TEXT_COLOR)
    
    # Remove bombs/group scoring overlay (score rendered in-frame via JIT renderer)
    screen.blit(lives_text, ((WIDTH - 60) * scale, 10))
    screen.blit(level_text, ((WIDTH // 2 - 20) * scale, 10))
    # no bombs/group overlay
    
    # Update display
    pygame.display.flip()
    
    # Cap framerate if clock is provided
    if fps_clock:
        fps_clock.tick(60)

def get_human_action() -> chex.Array:
    """Records keyboard input and returns the corresponding action."""
    import pygame
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return Action.LEFTFIRE
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return Action.RIGHTFIRE
    elif keys[pygame.K_LEFT]:
        return Action.LEFT
    elif keys[pygame.K_RIGHT]:
        return Action.RIGHT
    elif keys[pygame.K_SPACE]:
        return Action.FIRE
    else:
        return Action.NOOP

class KaboomRenderer(JAXGameRenderer):
    """Renderer for the Kaboom game."""
    
    def __init__(self, consts: KaboomConstants = None):
        super().__init__(consts)
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KaboomState) -> jnp.ndarray:
        """Render the game state."""
        return render_frame(state)

if __name__ == "__main__":
    import pygame
    import time
    import numpy as np
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * 3, HEIGHT * 3))
    pygame.display.set_caption("JAX Kaboom")
    
    # Initialize game
    game = JaxKaboom()
    
    # Game loop
    running = True
    key = jax.random.PRNGKey(0)
    key, state = game.reset_env(key)
    
    clock = pygame.time.Clock()
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get action
        action = get_human_action()
        
        # Update game state
        key, state, reward, done, info = game.step_env(key, state, action)
        
        # Display game (non-jitted function for display only)
        display_game(screen, state, clock)
        
        if done:
            print(f"Game over! Final score: {int(state.score)}")
            key, state = game.reset_env(key)
    
    pygame.quit()