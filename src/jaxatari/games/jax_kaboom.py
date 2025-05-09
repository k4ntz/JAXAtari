from functools import partial
from typing import NamedTuple, Tuple, Dict, List
import jax
import jax.numpy as jnp
import chex
from gymnax.environments import spaces
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

# Constants for game environment
WIDTH = 160
HEIGHT = 210
MAX_BOMBS = 8  # Maximum number of active bombs
MAX_STEPS = 10000

# Player (bucket) settings
BUCKET_WIDTH = 8
BUCKET_HEIGHT = 6
BUCKET_START_X = WIDTH // 2
BUCKET_START_Y = HEIGHT - 30  # Changed from HEIGHT - 70 to HEIGHT - 30
BUCKET_SPEED = 4

# Bomber (mad bomber) settings
BOMBER_WIDTH = 8
BOMBER_HEIGHT = 8
BOMBER_START_X = WIDTH // 2
BOMBER_START_Y = 30
BOMBER_MIN_X = 20
BOMBER_MAX_X = WIDTH - 20
BOMBER_SPEED = 2

# Bomb settings
BOMB_WIDTH = 4
BOMB_HEIGHT = 4
BOMB_SPEED_INITIAL = 1
BOMB_SPEED_MAX = 4
BOMB_SPEED_INCREMENT = 0.1  # Speed increases as game progresses

# Game settings
INITIAL_LIVES = 3
POINTS_PER_CATCH = 10

# Colors
BACKGROUND_COLOR = (0, 0, 0)
BUCKET_COLOR = (255, 255, 0)  # Yellow
BOMBER_COLOR = (255, 0, 0)    # Red
BOMB_COLOR = (255, 165, 0)    # Orange
TEXT_COLOR = (255, 255, 255)  # White

# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5

# State translator for debugging and visualization
STATE_TRANSLATOR: Dict[int, str] = {
    0: "bucket_x",
    1: "bomber_x",
    2: "bomber_direction",
    3: "score",
    4: "lives",
    5: "level",
    6: "step_counter",
    # For each possible bomb (up to MAX_BOMBS):
    # active, x, y, speed
}

# Dynamically add bomb state entries to STATE_TRANSLATOR
for i in range(MAX_BOMBS):
    base_idx = 7 + i * 4
    STATE_TRANSLATOR[base_idx] = f"bomb_{i}_active"
    STATE_TRANSLATOR[base_idx + 1] = f"bomb_{i}_x"
    STATE_TRANSLATOR[base_idx + 2] = f"bomb_{i}_y"
    STATE_TRANSLATOR[base_idx + 3] = f"bomb_{i}_speed"

# Immutable state container
class KaboomState(NamedTuple):
    bucket_x: chex.Array          # Bucket x-position
    bomber_x: chex.Array          # Bomber x-position
    bomber_direction: chex.Array  # Direction bomber is moving (-1: left, 1: right)
    score: chex.Array             # Current score
    lives: chex.Array             # Remaining lives
    level: chex.Array             # Current level (difficulty increases with level)
    step_counter: chex.Array      # Current step count
    bomb_active: chex.Array       # Boolean array tracking which bombs are active
    bomb_x: chex.Array            # X positions of all bombs
    bomb_y: chex.Array            # Y positions of all bombs
    bomb_speed: chex.Array        # Speed of each bomb
    drop_timer: chex.Array        # Timer for dropping new bombs
    obs_stack: chex.ArrayTree     # Observation stack for rendering

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class KaboomObservation(NamedTuple):
    bucket: EntityPosition
    bomber: EntityPosition
    bombs: List[EntityPosition]
    score: jnp.ndarray
    lives: jnp.ndarray
    level: jnp.ndarray

class KaboomInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array
    lives: jnp.ndarray
    score: jnp.ndarray
    level: jnp.ndarray

@jax.jit
def bucket_step(bucket_x: chex.Array, action: chex.Array) -> chex.Array:
    """Update bucket position based on action."""
    # Moving left
    bucket_x = jnp.where(
        jnp.logical_or(action == LEFT, action == LEFTFIRE),
        bucket_x - BUCKET_SPEED,
        bucket_x
    )
    
    # Moving right
    bucket_x = jnp.where(
        jnp.logical_or(action == RIGHT, action == RIGHTFIRE),
        bucket_x + BUCKET_SPEED,
        bucket_x
    )
    
    # Clamp bucket position to screen bounds
    bucket_x = jnp.clip(bucket_x, 0, WIDTH - BUCKET_WIDTH)
    
    return bucket_x

@jax.jit
def bomber_step(bomber_x: chex.Array, bomber_direction: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Update bomber position and direction."""
    # Move bomber based on current direction
    new_bomber_x = bomber_x + (bomber_direction * BOMBER_SPEED)
    
    # Check if bomber reached screen edge and needs to change direction
    hit_left_edge = new_bomber_x <= BOMBER_MIN_X
    hit_right_edge = new_bomber_x >= BOMBER_MAX_X
    hit_edge = jnp.logical_or(hit_left_edge, hit_right_edge)
    
    # Reverse direction if hit edge
    new_bomber_direction = jnp.where(
        hit_edge,
        -bomber_direction,
        bomber_direction
    )
    
    # Clamp bomber position
    new_bomber_x = jnp.clip(new_bomber_x, BOMBER_MIN_X, BOMBER_MAX_X)
    
    return new_bomber_x, new_bomber_direction

@jax.jit
def drop_bomb(
    bomb_active: chex.Array,
    bomb_x: chex.Array,
    bomb_y: chex.Array,
    bomb_speed: chex.Array,
    bomber_x: chex.Array,
    level: chex.Array,
    drop_timer: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Potentially drop a new bomb."""
    # Decrement drop timer
    new_drop_timer = drop_timer - 1
    
    # Check if it's time to drop a bomb
    should_drop = new_drop_timer <= 0
    
    # Find first inactive bomb slot
    inactive_slots = jnp.where(bomb_active == 0, 1, 0)
    first_inactive = jnp.argmax(inactive_slots)
    
    # Determine if we can and should drop a bomb
    can_drop = jnp.sum(bomb_active) < MAX_BOMBS
    will_drop = jnp.logical_and(should_drop, can_drop)
    
    # Reset timer if dropping
    new_drop_timer = jnp.where(
        will_drop,
        # Drop timer decreases with level (bombs drop more frequently)
        jnp.maximum(20 - level * 2, 5),
        new_drop_timer
    )
    
    # Calculate base speed based on level
    base_speed = jnp.minimum(BOMB_SPEED_INITIAL + level * BOMB_SPEED_INCREMENT, BOMB_SPEED_MAX)
    
    # Update bomb arrays if dropping
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
        jnp.where(will_drop, base_speed, bomb_speed[first_inactive])
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
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Update positions of active bombs and check for catches/misses."""
    
    # Move active bombs down
    new_bomb_y = jnp.where(
        bomb_active == 1,
        bomb_y + bomb_speed,
        bomb_y
    )
    
    # Common bucket properties for all buckets (same width and horizontal position)
    bucket_left = bucket_x
    bucket_right = bucket_x + BUCKET_WIDTH
    
    # Initialize a tensor to track if each bomb is caught by any bucket
    caught = jnp.zeros_like(bomb_active, dtype=jnp.bool_)
    
    # Check for catches with each visible bucket (player bucket + extra life buckets)
    def check_bucket_collision(bucket_index, caught_bombs):
        # Calculate the Y position of this bucket
        # bucket_index 0 = player bucket, 1 = first extra bucket, 2 = second extra bucket
        spacing = 5
        
        # For player bucket (index 0), use BUCKET_START_Y
        # For life buckets (1 and 2), calculate position below player bucket
        is_player_bucket = bucket_index == 0
        
        bucket_top = jnp.where(
            is_player_bucket,
            BUCKET_START_Y,
            BUCKET_START_Y + BUCKET_HEIGHT + spacing + ((bucket_index - 1) * (BUCKET_HEIGHT + spacing))
        )
        
        bucket_bottom = bucket_top + BUCKET_HEIGHT
        
        # Only check collision if this bucket should be visible
        # Player bucket (0) is visible if lives >= 1
        # First extra bucket (1) is visible if lives >= 2
        # Second extra bucket (2) is visible if lives >= 3
        bucket_visible = bucket_index < lives
        
        # Check collision with this bucket
        bucket_collision = jnp.logical_and(
            bomb_active == 1,
            jnp.logical_and(
                jnp.logical_and(
                    new_bomb_y + BOMB_HEIGHT >= bucket_top,
                    new_bomb_y <= bucket_bottom
                ),
                jnp.logical_and(
                    bomb_x + BOMB_WIDTH >= bucket_left,
                    bomb_x <= bucket_right
                )
            )
        )
        
        # Only count catches for buckets that should be visible
        valid_catches = jnp.logical_and(bucket_collision, bucket_visible)
        
        # Update the caught bombs array
        return jnp.logical_or(caught_bombs, valid_catches)
    
    # Check collisions with all possible buckets (player + up to 2 life buckets)
    caught = jax.lax.fori_loop(0, INITIAL_LIVES, check_bucket_collision, caught)
    
    # Check for misses - bombs that fall off screen
    missed = jnp.logical_and(
        bomb_active == 1,
        new_bomb_y > HEIGHT
    )
    
    # Update bomb active status (deactivate caught or missed bombs)
    new_bomb_active = jnp.where(
        jnp.logical_or(caught, missed),
        0,
        bomb_active
    )
    
    # Calculate rewards and update score
    catches = jnp.sum(caught)
    new_score = score + catches * POINTS_PER_CATCH
    
    # Update lives based on misses
    misses = jnp.sum(missed)
    new_lives = lives - misses
    
    # Update level based on score milestones
    new_level = jnp.floor(new_score / 100) + 1
    
    return new_bomb_active, bomb_x, new_bomb_y, bomb_speed, new_score, new_lives, new_level

@jax.jit
def step_fn(state: KaboomState, action: chex.Array) -> Tuple[KaboomState, chex.Array, chex.Array, KaboomInfo]:
    """Advance the game state by one step."""
    # Update bucket position based on action
    new_bucket_x = bucket_step(state.bucket_x, action)
    
    # Update bomber position and direction
    new_bomber_x, new_bomber_direction = bomber_step(state.bomber_x, state.bomber_direction)
    
    # Try dropping a new bomb
    new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed, new_drop_timer = drop_bomb(
        state.bomb_active, state.bomb_x, state.bomb_y, state.bomb_speed,
        new_bomber_x, state.level, state.drop_timer
    )
    
    # Update all active bombs and check for catches/misses
    new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed, new_score, new_lives, new_level = update_bombs(
        new_bomb_active, new_bomb_x, new_bomb_y, new_bomb_speed,
        new_bucket_x, state.score, state.lives, state.level
    )
    
    # Calculate reward (points gained in this step)
    reward = new_score - state.score
    
    # Check if game is over (no lives left)
    done = new_lives <= 0
    
    # Create updated state
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
        obs_stack=state.obs_stack  # Will be updated in the environment class
    )
    
    # Create info struct
    info = KaboomInfo(
        time=state.step_counter,
        all_rewards=reward,
        lives=new_lives,
        score=new_score,
        level=new_level
    )
    
    return new_state, reward, done, info

@jax.jit
def reset_fn(key: chex.PRNGKey) -> KaboomState:
    """Initialize a new game state."""
    key, subkey = jax.random.split(key)
    
    # Initialize with random bomber direction
    bomber_direction = jax.random.choice(subkey, jnp.array([-1, 1]))
    
    # Initialize arrays for bombs
    bomb_active = jnp.zeros(MAX_BOMBS, dtype=jnp.int32)
    bomb_x = jnp.zeros(MAX_BOMBS, dtype=jnp.float32)
    bomb_y = jnp.zeros(MAX_BOMBS, dtype=jnp.float32)
    bomb_speed = jnp.ones(MAX_BOMBS, dtype=jnp.float32) * BOMB_SPEED_INITIAL
    
    # Create initial state
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
        drop_timer=jnp.array(10, dtype=jnp.int32),  # Initial timer for first bomb drop
        obs_stack=None  # Will be initialized in the environment class
    )
    
    return state

class JaxKaboom(JaxEnvironment[KaboomState, KaboomObservation, KaboomInfo]):
    """JAX implementation of the Kaboom game environment."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def default_state(self) -> KaboomState:
        """Get the default initial state."""
        return reset_fn(jax.random.PRNGKey(0))
    
    def step_env(
        self, key: chex.PRNGKey, state: KaboomState, action: chex.Array
    ) -> Tuple[chex.Array, KaboomState, chex.Array, chex.Array, KaboomInfo]:
        """Take a step in the environment."""
        next_state, reward, done, info = step_fn(state, action)
        return key, next_state, reward, done, info
    
    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, KaboomState]:
        """Reset the environment."""
        key, subkey = jax.random.split(key)
        next_state = reset_fn(subkey)
        return key, next_state
    
    def get_obs(self, state: KaboomState) -> KaboomObservation:
        """Get observation from state."""
        # Convert state to observation
        bucket = EntityPosition(
            x=state.bucket_x,
            y=jnp.array(BUCKET_START_Y, dtype=jnp.float32),
            width=jnp.array(BUCKET_WIDTH, dtype=jnp.float32),
            height=jnp.array(BUCKET_HEIGHT, dtype=jnp.float32)
        )
        
        bomber = EntityPosition(
            x=state.bomber_x,
            y=jnp.array(BOMBER_START_Y, dtype=jnp.float32),
            width=jnp.array(BOMBER_WIDTH, dtype=jnp.float32),
            height=jnp.array(BOMBER_HEIGHT, dtype=jnp.float32)
        )
        
        bombs = []
        for i in range(MAX_BOMBS):
            bombs.append(EntityPosition(
                x=state.bomb_x[i],
                y=state.bomb_y[i],
                width=jnp.array(BOMB_WIDTH, dtype=jnp.float32),
                height=jnp.array(BOMB_HEIGHT, dtype=jnp.float32)
            ))
        
        return KaboomObservation(
            bucket=bucket,
            bomber=bomber,
            bombs=bombs,
            score=state.score,
            lives=state.lives,
            level=state.level
        )
    
    @property
    def num_actions(self) -> int:
        """Number of possible actions."""
        return 6  # NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    
    def action_space(self) -> spaces.Discrete:
        """Action space definition."""
        return spaces.Discrete(self.num_actions)
    
    def observation_space(self) -> spaces.Box:
        """Observation space definition."""
        return spaces.Box(0, 255, (HEIGHT, WIDTH, 3), dtype=jnp.uint8)
    
    def is_terminal(self, state: KaboomState) -> chex.Array:
        """Check if state is terminal."""
        return jnp.logical_or(
            state.lives <= 0,
            state.step_counter >= MAX_STEPS
        )

    def render(self, state: KaboomState) -> jnp.ndarray:
        """Render the current state as an RGB array."""
        # Create empty frame
        frame = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)
        
        # Draw bucket
        bucket_slice = jax.lax.dynamic_slice(
            frame,
            (int(BUCKET_START_Y), int(state.bucket_x), 0),
            (BUCKET_HEIGHT, BUCKET_WIDTH, 3)
        )
        bucket_color = jnp.array(BUCKET_COLOR, dtype=jnp.uint8)
        bucket_colored = jnp.ones_like(bucket_slice) * bucket_color
        frame = frame.at[
            BUCKET_START_Y:BUCKET_START_Y+BUCKET_HEIGHT,
            int(state.bucket_x):int(state.bucket_x)+BUCKET_WIDTH
        ].set(bucket_colored)
        
        # Draw bomber
        bomber_slice = jax.lax.dynamic_slice(
            frame,
            (BOMBER_START_Y, int(state.bomber_x), 0),
            (BOMBER_HEIGHT, BOMBER_WIDTH, 3)
        )
        bomber_color = jnp.array(BOMBER_COLOR, dtype=jnp.uint8)
        bomber_colored = jnp.ones_like(bomber_slice) * bomber_color
        frame = frame.at[
            BOMBER_START_Y:BOMBER_START_Y+BOMBER_HEIGHT,
            int(state.bomber_x):int(state.bomber_x)+BOMBER_WIDTH
        ].set(bomber_colored)
        
        # Draw active bombs
        def draw_bomb(frame, i):
            is_active = state.bomb_active[i] == 1
            bomb_x = int(state.bomb_x[i])
            bomb_y = int(state.bomb_y[i])
            
            # Only draw if bomb is active and within screen bounds
            valid_coords = (
                is_active &
                (bomb_x >= 0) &
                (bomb_x + BOMB_WIDTH <= WIDTH) &
                (bomb_y >= 0) &
                (bomb_y + BOMB_HEIGHT <= HEIGHT)
            )
            
            # Use lax.cond for conditional rendering
            frame = jax.lax.cond(
                valid_coords,
                lambda f: f.at[
                    bomb_y:bomb_y+BOMB_HEIGHT,
                    bomb_x:bomb_x+BOMB_WIDTH
                ].set(jnp.ones((BOMB_HEIGHT, BOMB_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BOMB_COLOR, dtype=jnp.uint8)),
                lambda f: f,
                frame
            )
            
            return frame
        
        # Draw all bombs using fori_loop instead of python for-loop
        frame = jax.lax.fori_loop(0, MAX_BOMBS, lambda i, f: draw_bomb(f, i), frame)
        
        # Draw score and lives (simplified - would be enhanced in a real implementation)
        # In a complete implementation, you would render text for score, lives, and level
        
        return frame

@jax.jit
def render_frame(state: KaboomState) -> jnp.ndarray:
    """Render the current state as an RGB array (JIT-compatible)."""
    # Create empty frame
    frame = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)
    
    # Draw bucket using dynamic_update_slice
    bucket_x = state.bucket_x.astype(jnp.int32)
    bucket_y = BUCKET_START_Y
    
    # Create a colored rectangle for the bucket
    bucket_rect = jnp.ones((BUCKET_HEIGHT, BUCKET_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BUCKET_COLOR, dtype=jnp.uint8)
    
    # Use dynamic_update_slice to place the bucket on the frame
    frame = jax.lax.dynamic_update_slice(frame, bucket_rect, (bucket_y, bucket_x, 0))
    
    # Draw bomber using dynamic_update_slice
    bomber_x = state.bomber_x.astype(jnp.int32)
    bomber_y = BOMBER_START_Y
    
    # Create a colored rectangle for the bomber
    bomber_rect = jnp.ones((BOMBER_HEIGHT, BOMBER_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BOMBER_COLOR, dtype=jnp.uint8)
    
    # Use dynamic_update_slice to place the bomber on the frame
    frame = jax.lax.dynamic_update_slice(frame, bomber_rect, (bomber_y, bomber_x, 0))
    
    # Draw bombs using fori_loop and dynamic_update_slice
    def draw_bomb(i, f):
        bomb_x = state.bomb_x[i].astype(jnp.int32)
        bomb_y = state.bomb_y[i].astype(jnp.int32)
        is_active = state.bomb_active[i] == 1
        
        # Create colored rectangle for the bomb
        bomb_rect = jnp.ones((BOMB_HEIGHT, BOMB_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BOMB_COLOR, dtype=jnp.uint8)
        
        # Only draw if bomb is active and within screen bounds
        valid_coords = (
            is_active &
            (bomb_x >= 0) &
            (bomb_x + BOMB_WIDTH <= WIDTH) &
            (bomb_y >= 0) &
            (bomb_y + BOMB_HEIGHT <= HEIGHT)
        )
        
        # Use lax.cond for conditional rendering
        f = jax.lax.cond(
            valid_coords,
            lambda _: jax.lax.dynamic_update_slice(f, bomb_rect, (bomb_y, bomb_x, 0)),
            lambda _: f,
            operand=None
        )
        
        return f
    
    # Draw all bombs
    frame = jax.lax.fori_loop(0, MAX_BOMBS, draw_bomb, frame)
    
    # Draw a simple score indicator (a line at the top)
    score_band = jnp.ones((8, WIDTH, 3), dtype=jnp.uint8) * jnp.array(TEXT_COLOR, dtype=jnp.uint8)
    frame = jax.lax.dynamic_update_slice(frame, score_band, (0, 0, 0))
    
    # Draw lives as vertical buckets below the player
    # They follow the player's horizontal position
    
    # Define life bucket dimensions
    LIFE_BUCKET_HEIGHT = BUCKET_HEIGHT
    LIFE_BUCKET_WIDTH = BUCKET_WIDTH
    LIFE_SPACING = 5  # Consistent spacing between all buckets
    
    # The main player bucket is considered the top-most life bucket
    # So we only need to draw additional buckets for lives > 1
    
    def draw_extra_life(i, f):
        # i=0 is the middle bucket (second life)
        # i=1 is the bottom bucket (third life)
        
        # Only draw if player has enough lives (i+2 because player bucket is the first life)
        life_number = i + 2  # life 2 or 3 (player bucket is life 1)
        has_life = life_number <= state.lives
        
        # Position lives vertically BELOW the player bucket, with same x-position
        life_x = state.bucket_x.astype(jnp.int32)  # Align with player bucket
        
        # Consistent spacing between all buckets
        spacing = 5
        
        # Start from the position immediately below the player bucket
        # Each subsequent bucket is positioned with consistent spacing
        life_y = BUCKET_START_Y + BUCKET_HEIGHT + spacing + (i * (LIFE_BUCKET_HEIGHT + spacing))
        
        # Create a bucket for each life
        life_rect = jnp.ones((LIFE_BUCKET_HEIGHT, LIFE_BUCKET_WIDTH, 3), dtype=jnp.uint8) * jnp.array(BUCKET_COLOR, dtype=jnp.uint8)
        
        # Use lax.cond for conditional rendering
        f = jax.lax.cond(
            has_life,
            lambda _: jax.lax.dynamic_update_slice(f, life_rect, (life_y, life_x, 0)),
            lambda _: f,
            operand=None
        )
        
        return f
    
    # Draw extra life indicators (up to 2 more buckets below the player bucket)
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
    
    # Add score text (can't do this in the jitted function)
    font = pygame.font.SysFont(None, 24)
    score_text = font.render(f"Score: {int(state.score)}", True, TEXT_COLOR)
    lives_text = font.render(f"Lives: {int(state.lives)}", True, TEXT_COLOR)
    level_text = font.render(f"Level: {int(state.level)}", True, TEXT_COLOR)
    
    screen.blit(score_text, (10 * scale, 10))
    screen.blit(lives_text, ((WIDTH - 60) * scale, 10))
    screen.blit(level_text, ((WIDTH // 2 - 20) * scale, 10))
    
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
        return LEFTFIRE
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return RIGHTFIRE
    elif keys[pygame.K_LEFT]:
        return LEFT
    elif keys[pygame.K_RIGHT]:
        return RIGHT
    elif keys[pygame.K_SPACE]:
        return FIRE
    else:
        return NOOP

class Renderer_AtraJaxisKaboom:
    """Renderer for the Kaboom game using AtraJaxis."""
    
    def __init__(self):
        # Fix the renderer initialization - use correct class name from the module
        self.aJ = aj.AtariJax(WIDTH, HEIGHT, 3)  # Changed from AtraJaxis to AtariJax
        self.score_font = None
    
    def render(self, state: KaboomState):
        """Render the game state."""
        self.aJ.fill(BACKGROUND_COLOR)
        
        # Draw bucket
        self.aJ.rect(
            int(state.bucket_x),
            BUCKET_START_Y,
            BUCKET_WIDTH,
            BUCKET_HEIGHT,
            BUCKET_COLOR
        )
        
        # Draw bomber
        self.aJ.rect(
            int(state.bomber_x),
            BOMBER_START_Y,
            BOMBER_WIDTH,
            BOMBER_HEIGHT,
            BOMBER_COLOR
        )
        
        # Draw active bombs
        for i in range(MAX_BOMBS):
            if state.bomb_active[i] == 1:
                self.aJ.rect(
                    int(state.bomb_x[i]),
                    int(state.bomb_y[i]),
                    BOMB_WIDTH,
                    BOMB_HEIGHT,
                    BOMB_COLOR
                )
        
        # Draw score
        self.aJ.text(
            f"Score: {int(state.score)}",
            10, 10,
            TEXT_COLOR
        )
        
        # Draw lives
        self.aJ.text(
            f"Lives: {int(state.lives)}",
            WIDTH - 60, 10,
            TEXT_COLOR
        )
        
        # Draw level
        self.aJ.text(
            f"Level: {int(state.level)}",
            WIDTH // 2 - 20, 10,
            TEXT_COLOR
        )
        
        self.aJ.update()

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