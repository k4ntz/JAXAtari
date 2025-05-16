import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, List, Union
import jax
import jax.numpy as jnp
import jax.random as random
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction
from jaxatari.renderers import AtraJaxisRenderer 

# Constants for game environment
WIDTH = 160
HEIGHT = 210
PLAYABLE_HEIGHT = 180 # Playable area excludes the bottom black bar

# Constants for player
PLAYER_WIDTH = 14
PLAYER_HEIGHT = 12
PLAYER_SPEED = 4
PLAYER_INITIAL_X = 80
PLAYER_INITIAL_Y = 140
PLAYER_COLOR = (169, 169, 169)

# Constants for buildings
# Constants for buildings
NUM_BUILDINGS = 3
BUILDING_WIDTH = 25
BUILDING_HEIGHT = 25
BUILDING_COLOR = (114, 114, 114)
MAX_BUILDING_DAMAGE = 14
BUILDING_INITIAL_Y = 160  # Changed to be lower, just above black bar
BUILDING_VELOCITY = 1  # pixels per frame
BUILDING_SPACING = 70

# Height and Y position based on damage level
BUILDING_HEIGHTS = jnp.array([225, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 0])

# Updated Y positions for buildings to sit directly on black bar
BUILDING_Y_POSITIONS = jnp.array([ 160, 163, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190])

# Constants for enemies
NUM_ENEMIES_PER_TYPE = 3
TOTAL_ENEMIES = NUM_ENEMIES_PER_TYPE * 4  # 4 types of enemies
ENEMY_INITIAL_Y = 69
ENEMY_SPEED = 1
ENEMY_SPAWN_Y = 30  # Initial Y position for newly spawned enemies
ENEMY_SPAWN_PROB = 0.02  # Probability to spawn a new enemy (SPEED) 

# Enemy types and their properties (width, height, RGB color, score value)
ENEMY_TYPES = {
    0: (16, 18, (135, 135, 135), 25),   # Enemy25
    1: (14, 16, (129, 129, 129), 50),   # Enemy50
    2: (14, 16, (86, 86, 186), 75),     # Enemy75
    3: (14, 14, (137, 137, 137), 100)   # Enemy100
}

# Constants for missiles
MISSILE_WIDTH = 2
MISSILE_HEIGHT = 2
MISSILE_COLOR = (236, 236, 236)
NUM_PLAYER_MISSILES = 1
NUM_ENEMY_MISSILES = 1
PLAYER_MISSILE_SPEED = -6 # Moving up is negative Y
ENEMY_MISSILE_SPEED = 4    # Moving down is positive Y
ENEMY_FIRE_PROB = 0.02     # Probability of an enemy firing per step

# Action constants
NOOP = JAXAtariAction.NOOP
FIRE = JAXAtariAction.FIRE
RIGHT = JAXAtariAction.RIGHT
LEFT = JAXAtariAction.LEFT
RIGHTFIRE = JAXAtariAction.RIGHTFIRE
LEFTFIRE = JAXAtariAction.LEFTFIRE

# Background color
BACKGROUND_COLOR = (144, 72, 17)

# Define the positions of the state information
STATE_TRANSLATOR: Dict[int, str] = {
    0: "player_x",
    1: "player_y",
    2: "player_lives", 
    # Further indices would map to other state elements
}

def get_human_action() -> chex.Array:
    """
    Records if LEFT, RIGHT, or FIRE is being pressed and returns the corresponding action.
    
    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    elif keys[pygame.K_LEFT]:
        return jnp.array(LEFT)
    elif keys[pygame.K_RIGHT]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)

# Immutable state container
class AirRaidState(NamedTuple):
    # Player state
    player_x: chex.Array
    player_y: chex.Array
    player_lives: chex.Array
    
    # Buildings state (3 buildings with damage levels)
    building_x: chex.Array
    building_y: chex.Array
    building_damage: chex.Array
    
    # Enemies state (4 types, 3 of each type)
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_active: chex.Array
    
    # Missiles state
    player_missile_x: chex.Array
    player_missile_y: chex.Array
    player_missile_active: chex.Array
    
    enemy_missile_x: chex.Array
    enemy_missile_y: chex.Array
    enemy_missile_active: chex.Array
    
    # Score and game state
    score: chex.Array
    step_counter: chex.Array
    rng: chex.Array  # Random key for stochastic game elements
    
    # Observation stack for RL
    obs_stack: chex.ArrayTree

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class AirRaidObservation(NamedTuple):
    player: EntityPosition
    buildings: List[EntityPosition]
    enemies: List[EntityPosition]
    player_missiles: List[EntityPosition]
    enemy_missiles: List[EntityPosition]
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
    # Check if left or right button was pressed
    move_left = jnp.logical_or(action == LEFT, action == LEFTFIRE)
    move_right = jnp.logical_or(action == RIGHT, action == RIGHTFIRE)
    
    # Calculate new position
    player_x = jnp.where(
        move_left,
        jnp.maximum(player_x - PLAYER_SPEED, 0),  # Move left with boundary check
        player_x
    )
    
    player_x = jnp.where(
        move_right,
        jnp.minimum(player_x + PLAYER_SPEED, WIDTH - PLAYER_WIDTH),  # Move right with boundary check
        player_x
    )
    
    return player_x

@jax.jit
def spawn_enemy(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Spawns a new enemy if conditions are met.
    
    Args:
        state: Current game state
        
    Returns:
        Updated enemy arrays (x, y, type, active, rng)
    """
    # Extract state components
    enemy_x = state.enemy_x
    enemy_y = state.enemy_y
    enemy_type = state.enemy_type
    enemy_active = state.enemy_active
    rng = state.rng
    
    # Generate random values for spawn decision and enemy properties
    rng, spawn_key = random.split(rng)
    spawn_prob = random.uniform(spawn_key)
    
    # Find the first inactive enemy
    inactive_mask = 1 - enemy_active
    inactive_indices = jnp.where(inactive_mask, jnp.arange(TOTAL_ENEMIES), -1)
    first_inactive = jnp.max(inactive_indices)  # Get the highest valid index
    
    # Randomize enemy type (0-3) and x position
    rng, type_key, pos_key = random.split(rng, 3)
    new_type = random.randint(type_key, shape=(), minval=0, maxval=4)  # 0-3 for enemy types
    new_x = random.randint(pos_key, shape=(), minval=10, maxval=WIDTH - 30)
    
    # Only spawn if probability is met and there's an inactive enemy slot
    should_spawn = jnp.logical_and(spawn_prob < ENEMY_SPAWN_PROB, first_inactive >= 0)
    
    # Update enemy state if spawning
    enemy_x = enemy_x.at[first_inactive].set(
        jnp.where(should_spawn, new_x, enemy_x[first_inactive])
    )
    enemy_y = enemy_y.at[first_inactive].set(
        jnp.where(should_spawn, ENEMY_SPAWN_Y, enemy_y[first_inactive])
    )
    enemy_type = enemy_type.at[first_inactive].set(
        jnp.where(should_spawn, new_type, enemy_type[first_inactive])
    )
    enemy_active = enemy_active.at[first_inactive].set(
        jnp.where(should_spawn, 1, enemy_active[first_inactive])
    )
    
    return enemy_x, enemy_y, enemy_type, enemy_active, rng

@jax.jit
def update_enemies(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Updates all enemy positions. Enemies move down the screen."""
    # Extract state components
    enemy_y = state.enemy_y
    enemy_active = state.enemy_active
    building_damage = state.building_damage
    
    # Move active enemies down
    enemy_y = jnp.where(enemy_active == 1, enemy_y + ENEMY_SPEED, enemy_y)
    
    # Deactivate enemies that reach the bottom
    reached_player = enemy_y > PLAYER_INITIAL_Y - 20  # Changed from HEIGHT to PLAYER_INITIAL_Y
    enemy_active = jnp.where(reached_player, 0, enemy_active)
    
    return enemy_y, enemy_active, building_damage


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
        jnp.logical_or(action == FIRE, action == LEFTFIRE),
        action == RIGHTFIRE
    )
    
    # Find the first inactive missile
    inactive_missile_mask = 1 - state.player_missile_active
    inactive_indices = jnp.where(inactive_missile_mask, jnp.arange(NUM_PLAYER_MISSILES), -1)
    first_inactive = jnp.max(inactive_indices)  # Get the highest valid index
    
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
def fire_enemy_missiles(state: AirRaidState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Randomly generates enemy missiles from active enemies.
    
    Args:
        state: Current game state
        
    Returns:
        Updated enemy missile positions, active flags, and RNG
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
    
    # This function finds a valid active enemy starting from random_idx
    def find_active_enemy(random_idx, enemy_active):
        # Create a shifted array where we start checking from random_idx
        indices = (random_idx + jnp.arange(TOTAL_ENEMIES)) % TOTAL_ENEMIES
        
        # For each index, check if it's active and compute a score
        # The first active enemy will have the highest score
        scores = jnp.where(
            enemy_active[indices] == 1, 
            TOTAL_ENEMIES - jnp.arange(TOTAL_ENEMIES), 
            -1
        )
        
        # Find the index with the highest score (first active enemy)
        best_idx = indices[jnp.argmax(scores)]
        
        # Return the best enemy index, or 0 if none found
        return jnp.where(jnp.max(scores) >= 0, best_idx, 0)
    
    # Find a valid active enemy
    firing_enemy_idx = find_active_enemy(random_enemy_idx, state.enemy_active)
    
    # Only fire if probability is met, enemy is available and there's an inactive missile slot
    enemy_available = active_enemy_count > 0
    can_fire = jnp.logical_and(
        jnp.logical_and(fire_prob < ENEMY_FIRE_PROB, first_inactive >= 0),
        enemy_available
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
    
    return enemy_missile_x, enemy_missile_y, enemy_missile_active, rng

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
            
            # Update state on collision
            enemy_active = enemy_active.at[e].set(
                jnp.where(effective_collision, 0, enemy_active[e])
            )
            
            player_missile_active = player_missile_active.at[pm].set(
                jnp.where(effective_collision, 0, player_missile_active[pm])
            )
            
            # Award score based on enemy type
            score_values = jnp.array([25, 50, 75, 100])
            score_to_add = score_values[state.enemy_type[e]]
            score = jnp.where(effective_collision, score + score_to_add, score)

    # Check enemy missiles hitting buildings and player
    for em in range(NUM_ENEMY_MISSILES):
        is_missile_active = enemy_missile_active[em]
        
        # First check collision with buildings
        for b in range(NUM_BUILDINGS):
            # Check collision between missile and building
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
            
            # Only count collision if missile is active
            effective_collision = jnp.logical_and(collision, is_missile_active == 1)
            
            # Update building damage and deactivate missile on collision
            building_damage = building_damage.at[b].set(
                jnp.where(effective_collision, 
                         jnp.minimum(building_damage[b] + 1, MAX_BUILDING_DAMAGE),
                         building_damage[b])
            )
            enemy_missile_active = enemy_missile_active.at[em].set(
                jnp.where(effective_collision, 0, enemy_missile_active[em])
            )
        
        # Then check collision with player
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
        
        # Only count collision if missile is active
        effective_player_collision = jnp.logical_and(player_collision, is_missile_active == 1)
        
        # Update state on collision
        enemy_missile_active = enemy_missile_active.at[em].set(
            jnp.where(effective_player_collision, 0, enemy_missile_active[em])
        )
        
        # Reduce player lives
        player_lives = jnp.where(effective_player_collision, player_lives - 1, player_lives)

    return enemy_active, player_missile_active, enemy_missile_active, score, player_lives, building_damage


class JaxAirRaid(JaxEnvironment[AirRaidState, AirRaidObservation, AirRaidInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list = None):
        super().__init__()
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        if reward_funcs is not None:
            self.reward_funcs = tuple(reward_funcs)
        else:
            self.reward_funcs = None
        self.action_set = {
            NOOP,
            FIRE,
            RIGHT,
            LEFT,
            RIGHTFIRE,
            LEFTFIRE
        }
        self.obs_size = 3*4+1+1  # Similar to Pong observation size
        self.renderer = AirRaidRenderer()  # Add renderer instance
    
    def get_renderer(self):
        # Returns the game's renderer.
        return self.renderer


    def get_action_space(self) -> chex.Array:
        return jnp.array([
            NOOP,
            FIRE,
            RIGHT,
            LEFT,
            RIGHTFIRE,
            LEFTFIRE
        ], dtype=jnp.int32)
    
    
    def reset(self, key=None) -> Tuple[AirRaidObservation, AirRaidState]:
        """
        Resets the game state to the initial state.
        
        Returns:
            The initial observation and state
        """
        # Initialize building positions
        building_x = jnp.array([
                -BUILDING_WIDTH,                    # First building
                -BUILDING_WIDTH + BUILDING_SPACING, # Second building
                -BUILDING_WIDTH + BUILDING_SPACING * 2  # Third building
        ])        
        building_y = jnp.array([BUILDING_INITIAL_Y, BUILDING_INITIAL_Y, BUILDING_INITIAL_Y])
        building_damage = jnp.zeros(NUM_BUILDINGS, dtype=jnp.int32)
        
        # Initialize enemy arrays (all inactive initially)
        enemy_x = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_y = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_type = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        enemy_active = jnp.zeros(TOTAL_ENEMIES, dtype=jnp.int32)
        
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
            player_missile_x=player_missile_x,
            player_missile_y=player_missile_y,
            player_missile_active=player_missile_active,
            enemy_missile_x=enemy_missile_x,
            enemy_missile_y=enemy_missile_y,
            enemy_missile_active=enemy_missile_active,
            score=jnp.array(0),
            step_counter=jnp.array(0),
            rng=rng,
            obs_stack=None # obs_stack will be initialized below
        )
        
        initial_obs = self._get_observation(state)
        
        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
        
        # Apply transformation to each leaf in the pytree
        initial_obs_stack = jax.tree.map(expand_and_copy, initial_obs)
        
        new_state_with_obs_stack = state._replace(obs_stack=initial_obs_stack)
        
        # Return (observation, state) - This is the crucial change
        return initial_obs, new_state_with_obs_stack

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AirRaidState, action: chex.Array) -> Tuple[AirRaidObservation, AirRaidState, float, bool, AirRaidInfo]:
        """
        Steps the game state forward by one frame.
        
        Args:
            state: Current game state
            action: Action to take
            
        Returns:
            Updated game state, observation, reward, done flag, and info
        """
        # Always assume state is AirRaidState - remove all type checking code
        
        # Update building positions
        new_building_x = state.building_x + BUILDING_VELOCITY
        new_building_x = jnp.where(
            new_building_x > WIDTH,
            new_building_x - (WIDTH + BUILDING_WIDTH + BUILDING_SPACING),  # Move back by screen width plus spacing
            new_building_x
        )

        # Continue with the rest of the step logic...
        # 1. Update player position
        new_player_x = player_step(state.player_x, action)
        
        # 2. Update buildings (they don't move, just store their state)
        
        # 3. Spawn new enemies
        new_enemy_x, new_enemy_y, new_enemy_type, new_enemy_active, new_rng = spawn_enemy(state._replace(player_x=new_player_x))
        
        # 4. Update existing enemies
        updated_enemy_y, updated_enemy_active, updated_building_damage = update_enemies(
            state._replace(
                player_x=new_player_x,
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_type=new_enemy_type,
                enemy_active=new_enemy_active,
                rng=new_rng
            )
        )
        
        # 5. Handle player firing missiles
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
        
        # 6. Handle enemy firing missiles
        new_enemy_missile_x, new_enemy_missile_y, new_enemy_missile_active, newer_rng = fire_enemy_missiles(
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
                rng=new_rng
            )
        )
        
        # 7. Update missile positions
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
        
        # 8. Detect and handle collisions
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
            enemy_missile_active=updated_enemy_missile_active
        )
        )
        
        # 9. Create the new state
        new_state = AirRaidState(
            player_x=new_player_x,
            player_y=state.player_y,  # Player y doesn't change in AirRaid
            player_lives=new_player_lives,
            building_x=new_building_x,
            building_y=state.building_y,
            building_damage=final_building_damage, 
            enemy_x=new_enemy_x,
            enemy_y=updated_enemy_y,
            enemy_type=new_enemy_type,
            enemy_active=final_enemy_active,
            player_missile_x=new_player_missile_x,
            player_missile_y=updated_player_missile_y,
            player_missile_active=final_player_missile_active,
            enemy_missile_x=new_enemy_missile_x,
            enemy_missile_y=updated_enemy_missile_y,
            enemy_missile_active=final_enemy_missile_active,
            score=new_score,
            step_counter=state.step_counter + 1,
            rng=newer_rng,
            obs_stack=state.obs_stack  # Will update this below
        )
        
        # 10. Get reward, done, and info
        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        
        # 11. Get observation
        observation = self._get_observation(new_state)
        
        # Update obs_stack for RL
        # Ensure new_state.obs_stack is not None before trying to tree_map over it.
        # This should be initialized correctly in reset.
        current_obs_stack = new_state.obs_stack
        
        # If obs_stack was None (e.g. if reset didn't initialize it properly for some reason, though it should)
        # We might need a fallback, but ideally, reset handles this.
        # For safety, let's assume obs_stack is correctly initialized by reset.

        observation_stack = jax.tree.map(
            lambda stack, obs_leaf: jnp.concatenate([stack[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0),
            current_obs_stack, # Use current_obs_stack from new_state
            observation
        )
        final_new_state = new_state._replace(obs_stack=observation_stack) # Use a different variable name to avoid confusion
        
        # Return (observation, new_state, reward, done, info) - Correct order for play.py
        def do_reset(_):
            obs, reset_state = self.reset(new_state.rng)
            return obs, reset_state, env_reward, False, info

        def no_reset(_):
            return observation, final_new_state, env_reward, done, info

        return jax.lax.cond(done, do_reset, no_reset, operand=None)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AirRaidState) -> AirRaidObservation:
        """
        Transforms the raw state into an observation.

        Args:
            state: Current game state

        Returns:
            Observation object containing entity positions and game data
        """
        # Create player entity
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_WIDTH),
            height=jnp.array(PLAYER_HEIGHT)
        )

        # Create building entities
        buildings = []
        for i in range(NUM_BUILDINGS):
            # Adjust height based on damage
            height = BUILDING_HEIGHTS[state.building_damage[i]]
            y_pos = BUILDING_Y_POSITIONS[state.building_damage[i]]

            buildings.append(EntityPosition(
                x=state.building_x[i],
                y=jnp.array(y_pos),
                width=jnp.array(BUILDING_WIDTH),
                height=jnp.array(height)
            ))

        # Create enemy entities - use JAX's where to conditionally set values
        enemies = []
        for i in range(TOTAL_ENEMIES):
            # Get dimensions based on enemy type
            width = jnp.select(
                [state.enemy_type[i] == 0, state.enemy_type[i] < 3, state.enemy_type[i] == 3],
                [16, 14, 14],
                default=0
            )

            height = jnp.select(
                [state.enemy_type[i] == 0, state.enemy_type[i] < 3, state.enemy_type[i] == 3],
                [18, 16, 14],
                default=0
            )

            # Use where to set x/y to 0 if inactive
            x = jnp.where(state.enemy_active[i] == 1, state.enemy_x[i], 0)
            y = jnp.where(state.enemy_active[i] == 1, state.enemy_y[i], 0)
            width = jnp.where(state.enemy_active[i] == 1, width, 0)
            height = jnp.where(state.enemy_active[i] == 1, height, 0)

            enemies.append(EntityPosition(
                x=x,
                y=y,
                width=width,
                height=height
            ))

        # Create player missile entities - similar approach
        player_missiles = []
        for i in range(NUM_PLAYER_MISSILES):
            x = jnp.where(state.player_missile_active[i] == 1, state.player_missile_x[i], 0)
            y = jnp.where(state.player_missile_active[i] == 1, state.player_missile_y[i], 0)
            width = jnp.where(state.player_missile_active[i] == 1, MISSILE_WIDTH, 0)
            height = jnp.where(state.player_missile_active[i] == 1, MISSILE_HEIGHT, 0)

            player_missiles.append(EntityPosition(
                x=x,
                y=y,
                width=jnp.array(width),
                height=jnp.array(height)
            ))

        # Create enemy missile entities
        enemy_missiles = []
        for i in range(NUM_ENEMY_MISSILES):
            x = jnp.where(state.enemy_missile_active[i] == 1, state.enemy_missile_x[i], 0)
            y = jnp.where(state.enemy_missile_active[i] == 1, state.enemy_missile_y[i], 0)
            width = jnp.where(state.enemy_missile_active[i] == 1, MISSILE_WIDTH, 0)
            height = jnp.where(state.enemy_missile_active[i] == 1, MISSILE_HEIGHT, 0)

            enemy_missiles.append(EntityPosition(
                x=x,
                y=y,
                width=jnp.array(width),
                height=jnp.array(height)
            ))

        return AirRaidObservation(
            player=player,
            buildings=buildings,
            enemies=enemies,
            player_missiles=player_missiles,
            enemy_missiles=enemy_missiles,
            score=state.score,
            lives=state.player_lives
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AirRaidObservation) -> jnp.ndarray:
        """
        Converts the observation to a flat array for RL algorithms.
        
        Args:
            obs: Observation object
            
        Returns:
            Flattened array representation of the observation
        """
        # Convert player data
        player_data = jnp.array([
            obs.player.x, 
            obs.player.y, 
            obs.player.width, 
            obs.player.height
        ])
        
        # Extract building data - first building only for simplicity
        building_data = jnp.array([
            obs.buildings[0].x,
            obs.buildings[0].y,
            obs.buildings[0].width,
            obs.buildings[0].height
        ])
        
        # Extract enemy data - first enemy only for simplicity
        enemy_data = jnp.array([
            obs.enemies[0].x,
            obs.enemies[0].y,
            obs.enemies[0].width,
            obs.enemies[0].height
        ])
        
        # Extract missile data - first missile only for simplicity
        player_missile_data = jnp.array([
            obs.player_missiles[0].x,
            obs.player_missiles[0].y,
            obs.player_missiles[0].width,
            obs.player_missiles[0].height
        ])
        
        enemy_missile_data = jnp.array([
            obs.enemy_missiles[0].x,
            obs.enemy_missiles[0].y,
            obs.enemy_missiles[0].width,
            obs.enemy_missiles[0].height
        ])
        
        # Combine all data
        return jnp.concatenate([
            player_data.flatten(),
            building_data.flatten(),
            enemy_data.flatten(),
            player_missile_data.flatten(),
            enemy_missile_data.flatten(),
            obs.score.flatten(),
            obs.lives.flatten()
        ])
    
    def action_space(self) -> spaces.Discrete:
        """
        Returns the action space for the environment.
        """
        return spaces.Discrete(len(self.action_set))
    
    def observation_space(self) -> spaces.Box:
        """
        Returns the observation space for the environment.
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AirRaidState, all_rewards: chex.Array) -> AirRaidInfo:
        """
        Returns additional info about the current game state.
        """
        return AirRaidInfo(time=state.step_counter, all_rewards=all_rewards)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: AirRaidState, state: AirRaidState) -> float:
        """
        Calculates the environment reward based on score increase and life loss.
        """
        score_reward = state.score - previous_state.score
        life_penalty = (previous_state.player_lives - state.player_lives) * 25
        return score_reward - life_penalty
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: AirRaidState, state: AirRaidState) -> chex.Array:
        """
        Calculates all custom rewards if reward functions are provided.
        """
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AirRaidState) -> bool:
        """
        Determines if the game is over.
        """
        # Game is over if player has no lives left
        return jnp.less_equal(state.player_lives, 0)

def load_sprites():
    """Load all sprites required for AirRaid rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load sprites (these would need to be created and saved as .npy files)
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/player.npy"), transpose=True)
    building = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/building.npy"), transpose=True)
    enemy25 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy25.npy"), transpose=True)
    enemy50 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy50.npy"), transpose=True)
    enemy75 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy75.npy"), transpose=True)
    enemy100 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/enemy100.npy"), transpose=True)
    missile = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/missile.npy"), transpose=True)
    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/background.npy"), transpose=True)
    life = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/airraid/life.npy"), transpose=True)
    
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
    DIGIT_SPRITES = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/airraid/score_{}.npy"),
        num_chars=11, # we added 1 so that we can have an invisible sprite
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

class AirRaidRenderer(AtraJaxisRenderer):
    """JAX-based AirRaid game renderer, optimized with JIT compilation."""
    
    def __init__(self):
        super().__init__()
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
     ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state_or_obs):
        """
        Renders the current game state using JAX operations.
        """
        # Determine if we're receiving a state or observation
        is_observation = isinstance(state_or_obs, AirRaidObservation)
        
        # Create empty raster with correct orientation for atraJaxis framework
        raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)

        # Render background
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Extract scalar values from possibly batched values if necessary
        def get_scalar(value):
            # Check if value has more than 0 dimensions and size > 1
            if hasattr(value, 'shape') and value.ndim > 0 and value.size > 0:
                # Take the first element if it's an array
                return value.item() if value.size == 1 else value[0]
            return value

        # Render buildings
        if is_observation:
            # For observation, handle buildings with static indexing
            # Use get_scalar to ensure we have simple integer coordinates
            if len(state_or_obs.buildings) > 0:
                frame_building = aj.get_sprite_frame(self.SPRITE_BUILDING, 0)
                building = state_or_obs.buildings[0]
                x = get_scalar(building.x)
                y = get_scalar(building.y)
                raster = aj.render_at(raster, x, y, frame_building)
                
            if len(state_or_obs.buildings) > 1:
                frame_building = aj.get_sprite_frame(self.SPRITE_BUILDING, 0)
                building = state_or_obs.buildings[1]
                x = get_scalar(building.x)
                y = get_scalar(building.y)
                raster = aj.render_at(raster, x, y, frame_building)
                
            if len(state_or_obs.buildings) > 2:
                frame_building = aj.get_sprite_frame(self.SPRITE_BUILDING, 0)
                building = state_or_obs.buildings[2]
                x = get_scalar(building.x)
                y = get_scalar(building.y)
                raster = aj.render_at(raster, x, y, frame_building)
        else:
            # For state, use the original detailed rendering with damage levels
            def render_building(i, raster_in):
                frame_building = aj.get_sprite_frame(self.SPRITE_BUILDING, 0)
                damage_level = state_or_obs.building_damage[i]
                
                # Get building position
                building_x = state_or_obs.building_x[i]
                building_y = BUILDING_Y_POSITIONS[damage_level]
                
                # Render building
                return aj.render_at(raster_in, building_x, building_y, frame_building)
       
            # Use loop to render all buildings
            raster = jax.lax.fori_loop(0, NUM_BUILDINGS, render_building, raster)

        # Render enemies
        if is_observation:
            # For observation, handle enemies with static indexing
            for i in range(min(len(state_or_obs.enemies), TOTAL_ENEMIES)):
                if i < len(state_or_obs.enemies):
                    enemy = state_or_obs.enemies[i]
                    
                    # Convert width/height to scalars
                    width = get_scalar(enemy.width)
                    height = get_scalar(enemy.height)
                    
                    # Create active condition as a scalar
                    active_cond = jnp.logical_and(width > 0, height > 0)
                    
                    # Pre-load all sprite types
                    sprite_25 = aj.get_sprite_frame(self.SPRITE_ENEMY25, 0)
                    sprite_50 = aj.get_sprite_frame(self.SPRITE_ENEMY50, 0)
                    sprite_100 = aj.get_sprite_frame(self.SPRITE_ENEMY100, 0)
                    
                    # Determine sprite type using jnp.where instead of if/else
                    is_type_25 = width == 16
                    is_type_50_75 = jnp.logical_and(width == 14, height == 16)
                    
                    sprite = jnp.where(
                        is_type_25, 
                        sprite_25,
                        jnp.where(
                            is_type_50_75,
                            sprite_50,
                            sprite_100
                        )
                    )
                    
                    # Use JAX conditional rendering
                    def render_enemy(unused):
                        x_pos = get_scalar(enemy.x)
                        y_pos = get_scalar(enemy.y)
                        return aj.render_at(raster, x_pos, y_pos, sprite)
                    
                    def no_render(unused):
                        return raster
                    
                    # Use lax.cond instead of if statement
                    raster = jax.lax.cond(
                        active_cond,
                        render_enemy,
                        no_render,
                        operand=None
                    )
        else:
            # Original enemy rendering for state
            def render_enemy(i, raster_in):
                # Create conditions for each enemy type
                is_active = state_or_obs.enemy_active[i] == 1
                is_type0 = state_or_obs.enemy_type[i] == 0
                is_type1 = state_or_obs.enemy_type[i] == 1
                is_type2 = state_or_obs.enemy_type[i] == 2
                is_type3 = state_or_obs.enemy_type[i] == 3
            
                # Pre-load all sprite types
                sprite_25 = aj.get_sprite_frame(self.SPRITE_ENEMY25, 0)
                sprite_50 = aj.get_sprite_frame(self.SPRITE_ENEMY50, 0)
                sprite_75 = aj.get_sprite_frame(self.SPRITE_ENEMY75, 0)
                sprite_100 = aj.get_sprite_frame(self.SPRITE_ENEMY100, 0)
            
                # Conditionally choose sprite - will only be used if active
                enemy_sprite = jnp.where(is_type0, sprite_25,
                            jnp.where(is_type1, sprite_50,
                            jnp.where(is_type2, sprite_75, sprite_100)))
            
                # Render only if active
                render_result = aj.render_at(raster_in, state_or_obs.enemy_x[i], state_or_obs.enemy_y[i], enemy_sprite)
                return jnp.where(is_active, render_result, raster_in)
            
            # Render all enemies
            raster = jax.lax.fori_loop(0, TOTAL_ENEMIES, render_enemy, raster)

        # Render player
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        if is_observation:
            x = get_scalar(state_or_obs.player.x)
            y = get_scalar(state_or_obs.player.y)
            raster = aj.render_at(raster, x, y, frame_player)
        else:
            raster = aj.render_at(raster, state_or_obs.player_x, state_or_obs.player_y, frame_player)

        # Render player missiles
        if is_observation:
            # For observation, use static indexing
            for i in range(min(len(state_or_obs.player_missiles), NUM_PLAYER_MISSILES)):
                if i < len(state_or_obs.player_missiles):
                    missile = state_or_obs.player_missiles[i]
                    
                    # Get scalar values and create condition
                    width = get_scalar(missile.width)
                    height = get_scalar(missile.height)
                    active_cond = jnp.logical_and(width > 0, height > 0)
                    
                    # Define render functions for conditional
                    def render_missile(unused):
                        x_pos = get_scalar(missile.x)
                        y_pos = get_scalar(missile.y)
                        frame_missile = aj.get_sprite_frame(self.SPRITE_MISSILE, 0)
                        return aj.render_at(raster, x_pos, y_pos, frame_missile)
                        
                    def no_render(unused):
                        return raster
                    
                    # Use lax.cond instead of if statement
                    raster = jax.lax.cond(
                        active_cond,
                        render_missile,
                        no_render,
                        operand=None
                    )
        else:
            def render_player_missile(i, raster_in):
                frame_missile = aj.get_sprite_frame(self.SPRITE_MISSILE, 0)
                render_result = aj.render_at(raster_in, state_or_obs.player_missile_x[i], 
                                            state_or_obs.player_missile_y[i], frame_missile)
                return jnp.where(state_or_obs.player_missile_active[i] == 1, render_result, raster_in)
        
            raster = jax.lax.fori_loop(0, NUM_PLAYER_MISSILES, render_player_missile, raster)

        # Render enemy missiles
        if is_observation:
            # For observation, use static indexing for enemy missiles too
            for i in range(min(len(state_or_obs.enemy_missiles), NUM_ENEMY_MISSILES)):
                if i < len(state_or_obs.enemy_missiles):
                    missile = state_or_obs.enemy_missiles[i]
                    
                    # Get scalar values and create condition
                    width = get_scalar(missile.width)
                    height = get_scalar(missile.height)
                    active_cond = jnp.logical_and(width > 0, height > 0)
                    
                    # Define render functions for conditional
                    def render_missile(unused):
                        x_pos = get_scalar(missile.x)
                        y_pos = get_scalar(missile.y)
                        frame_missile = aj.get_sprite_frame(self.SPRITE_MISSILE, 0)
                        return aj.render_at(raster, x_pos, y_pos, frame_missile)
                        
                    def no_render(unused):
                        return raster
                    
                    # Use lax.cond instead of if statement
                    raster = jax.lax.cond(
                        active_cond,
                        render_missile,
                        no_render,
                        operand=None
                    )
        else:
            def render_enemy_missile(i, raster_in):
                frame_missile = aj.get_sprite_frame(self.SPRITE_MISSILE, 0)
                render_result = aj.render_at(raster_in, state_or_obs.enemy_missile_x[i], 
                                            state_or_obs.enemy_missile_y[i], frame_missile)
                return jnp.where(state_or_obs.enemy_missile_active[i] == 1, render_result, raster_in)
        
            raster = jax.lax.fori_loop(0, NUM_ENEMY_MISSILES, render_enemy_missile, raster)

        # Add black bar at the bottom
        black_bar_height = 20
        black_bar_y = HEIGHT - black_bar_height
        raster = raster.at[:, black_bar_y:, :].set(0)  # Set to black (0,0,0)

        # Render score
        if is_observation:
            score_value = get_scalar(state_or_obs.score)
        else:
            score_value = (state_or_obs.score // 25) * 25
            
        score_y = 5 # Use a fixed Y position
        score_x_start = 30 # Fixed start position for 6 digits

        # Get digits padded to a fixed length (6)
        max_digits_render = 6
        padded_digits_render = jnp.zeros(max_digits_render, dtype=jnp.int32)

        # Use lax.fori_loop to extract digits
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
        indices = jnp.arange(max_digits_render) # [0, 1, 2, 3, 4, 5]

        # Find the index of the first significant digit (first non-zero)
        is_significant = (padded_digits_render > 0)
        # Assign a large index if a digit is not significant
        temp_indices = jnp.where(is_significant, indices, max_digits_render)
        # The minimum of these temp indices is the index of the first significant digit
        first_significant_idx = jnp.min(temp_indices)

        # Determine visibility mask
        visible_if_zero = (indices == max_digits_render - 1)
        visible_if_nonzero = (indices >= first_significant_idx)

        # Combine conditions
        should_be_visible = jnp.where(is_score_zero, visible_if_zero, visible_if_nonzero)

        # Select final digit indices to render
        invisible_digit_index = 10 # Transparent sprite index
        final_digits_to_render = jnp.where(
            should_be_visible,
            padded_digits_render,      # Use the actual calculated digit if visible
            invisible_digit_index      # Use the invisible index if not visible
        )

        # Render the score
        raster = aj.render_label(
            raster,
            score_x_start,
            score_y,
            final_digits_to_render,
            self.DIGIT_SPRITES
        )

        # Render lives
        def render_life(i, raster_in):
            life_sprite = aj.get_sprite_frame(self.SPRITE_LIFE, 0)
            life_width = life_sprite.shape[0]
            life_spacing = life_width + 3  # Add some spacing between lives

            # Position at 166px in x, 100px up from bottom
            life_start_x = 30
            life_y = 200  # 100px up from bottom

            # Calculate position
            icon_x = life_start_x + i * life_spacing

            # Render life sprite
            result = aj.render_at(raster_in, icon_x, life_y, life_sprite)

            # Only show the life icon if the player has enough lives
            if is_observation:
                lives = get_scalar(state_or_obs.lives)
                return jnp.where(i < lives, result, raster_in)
            else:
                return jnp.where(i < state_or_obs.player_lives, result, raster_in)

        raster = jax.lax.fori_loop(0, 5, render_life, raster)

        return raster

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    WINDOW_WIDTH = 160 * 3
    WINDOW_HEIGHT = 210 * 3
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Air Raid Game")
    clock = pygame.time.Clock()

    # Create the game instance
    game = JaxAirRaid(frameskip=1)

    # Create the JAX renderer
    renderer = AirRaidRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    # Initialize game state
    obs, curr_state = jitted_reset()

    # Game loop
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n: # Ensure it's KEYDOWN for 'n'
                if frame_by_frame: # Only step if in frame_by_frame mode
                    if counter % frameskip == 0: # Respect frameskip
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )
                        # Reset if game is done
                        if done:
                           obs, curr_state = jitted_reset()


        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)
                
                # Reset if game is done
                if done:
                    obs, curr_state = jitted_reset()
                   
        # Render and display
        raster = renderer.render(curr_state)

        # Update pygame display
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()
