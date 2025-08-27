import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

WIDTH = 160
HEIGHT = 210

NOOP = 5
FIRE = 4
LEFT = 3
RIGHT = 2
UP = 1
DOWN = 0

# Max Amount of Fuel, results in 170 seconds of time with speed of 1 and 60 fps
FUEL_CAPACITY = 10200.0
# Height of the Fuel Tank in pixels
TANK_HEIGHT = 25.0
# Levels to which the tank can be refilled
TANK_LEVELS = jnp.array([0, 1, 2, 4, 6, 9, 12, 16, 18, 25])
# Fuel Refill amounts by amount of banks cleared
REFILL_TABLE = TANK_LEVELS / TANK_HEIGHT * FUEL_CAPACITY
# Fuel Cost of using Dynamite
DYNAMITE_COST = FUEL_CAPACITY * 0.02
# Delay before Dynamite Explodes (1 second at 60fps)
DYNAMITE_DELAY = 60
# Fuel gained on revival after "dying" of lack of fuel
REVIVAL_FUEL = jnp.array(0.2 * FUEL_CAPACITY).astype(jnp.float32)
# Base Speed in the first difficulty level
BASE_SPEED = 0.5
# Speed Increase per level higher than 1
SPEED_INCREASE_PER_LEVEL = jnp.array(1.1)
# Fuel Consumption Increase per level higher than 1
FUEL_CONSUMPTION_INCREASE_PER_LEVEL = jnp.array(1.15)
# Number of cities to clear before level increases
CITIES_PER_LEVEL = 4
# Max Level
MAX_LEVEL = 8
# Max Lives
MAX_LIVES = 6
# Starting_Lives
STARTING_LIVES = 4
# Reward for robbing a bank
BANK_ROBBERY_REWARD = 10
# Reward for killing a police car with dynamite
POLICE_KILL_REWARD = 40
# Position of the Fuel_Tank Sprite
FUEL_TANK_POSITION = (42, 12)
# Position of the first life
FIRST_LIFE_POSITION = (WIDTH-70, 27)
# Offset of the lives first value is the x dimension offset second offset between rows
LIFE_OFFSET = (16, 12)
# Speed factor for police cars, police cars are slower than player
POLICE_SLOW_DOWN_FACTOR = 0.9

# Array containing position of all 6 lives calculated from FIRST_LIFE_POSITION and LIFE_OFFSET
LIFE_POSITIONS = jnp.array([
    (FIRST_LIFE_POSITION[0] + i%3 * LIFE_OFFSET[0], FIRST_LIFE_POSITION[1] - (i//3) * LIFE_OFFSET[1])
    for i in range(MAX_LIVES)
])

WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

COLLISION_BOX = (8, 8)
PORTAL_X = jnp.array([12, 140])

# Police spawn delay (120 frames = 2 seconds at 60 FPS)
POLICE_SPAWN_DELAY = 120

# Police AI bias factors
POLICE_RANDOM_FACTOR = 0.7  # 70% random movement
POLICE_BIAS_FACTOR = 0.5    # 50% bias towards player

DYNAMITE_EXPLOSION_DELAY = 30

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(MODULE_DIR, "sprites", "bankheist")

def init_banks_or_police() -> chex.Array:
    """
    Initializes the bank and police positions.

    Returns:
        chex.Array: An array containing the initial positions of banks and police.
    """
    positions = jnp.stack([jnp.array([0, 0]), jnp.array([0, 0]), jnp.array([0, 0])])
    directions = jnp.stack([jnp.array(4), jnp.array(4), jnp.array(4)])
    visibilities = jnp.stack([jnp.array(0), jnp.array(0), jnp.array(0)])
    return Entity(position=positions, direction=directions, visibility=visibilities)

def load_city_collision_map(file_name: str) -> chex.Array:
    """
    Loads the city collision map from the sprites directory.
    """
    map = jnp.load(os.path.join(SPRITES_DIR, file_name))
    map = map[..., 0].squeeze()
    return jnp.transpose(map, (1, 0))

def get_spawn_points(maps: chex.Array) -> chex.Array:
    spawn_maps = [find_free_areas(map, h=8, w=8) for map in maps]
    min_length = min(len(spawn_points) for spawn_points in spawn_maps)
    key = jax.random.PRNGKey(0)
    shuffled_spawn_maps = [jax.random.permutation(key,spawn_points)[:min_length] for spawn_points in spawn_maps]
    return jnp.stack(shuffled_spawn_maps, axis=0)

def find_free_areas(map, h, w):
    free_mask = (map == 0)
    H, W = free_mask.shape

    def check_window(i, j):
        window = jax.lax.dynamic_slice(free_mask, (i, j), (h, w))
        return jnp.all(window)

    # Generate all possible top-left positions
    rows = jnp.arange(H - h + 1)
    cols = jnp.arange(W - w + 1)
    # Create a grid of all possible positions
    grid_i, grid_j = jnp.meshgrid(rows, cols, indexing='ij')
    # Apply stride and offset using boolean masks
    row_mask = (grid_i % 8 == 4)
    col_mask = (grid_j % 8 == 5)
    mask = row_mask & col_mask
    # Get valid positions
    i_idx, j_idx = jnp.where(mask)
    positions = jnp.stack(jnp.array([i_idx, j_idx]).astype(jnp.int32), axis=-1)

    def scan_fn(carry, pos):
        i, j = pos
        is_free = check_window(i, j)
        return carry, is_free

    _, is_free_arr = jax.lax.scan(scan_fn, None, positions)
    valid_positions = positions[is_free_arr]
    return jnp.array(valid_positions)

CITY_COLLISION_MAPS = jnp.array([load_city_collision_map(f"map_{i+1}_collision.npy") for i in range(8)])
CITY_SPAWNS = get_spawn_points(CITY_COLLISION_MAPS)

def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_w] or keys[pygame.K_UP]:
        return jnp.array(UP)
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return jnp.array(DOWN)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    return jnp.array(NOOP)

class Entity(NamedTuple):
    position: chex.Array
    direction: chex.Array
    visibility: chex.Array

class BankHeistState(NamedTuple):
    level: chex.Array
    difficulty_level: chex.Array
    player: Entity
    dynamite_position: chex.Array
    enemy_positions: Entity
    bank_positions: Entity
    speed: chex.Array
    fuel_consumption: chex.Array
    reserve_speed: chex.Array # how much movement could not be used in the last tick
    police_reserve_speed: chex.Array # how much movement the police could not use in the last tick
    money: chex.Array
    player_lives: chex.Array
    fuel: chex.Array
    fuel_refill: chex.Array
    obs_stack: chex.ArrayTree
    map_collision: chex.Array
    spawn_points: chex.Array
    bank_spawn_timers: chex.Array
    police_spawn_timers: chex.Array
    dynamite_timer: chex.Array
    pending_police_spawns: chex.Array  # Timer for delayed police spawning
    pending_police_bank_indices: chex.Array  # Bank indices where police should spawn
    game_paused: chex.Array  # Game paused state Is set at beginning and after life was lost
    bank_heists: chex.Array  # number of bank heists completed in the current level
    pending_police_bank_indices: chex.Array  # Bank indices where police should spawn
    random_key: chex.PRNGKey  # Persistent random key that advances each step  

#TODO: Add Background collision Map, Fuel, Fuel Refill and others
class BankHeistObservation(NamedTuple):
    player: Entity
    lives: jnp.ndarray
    score: jnp.ndarray
    enemies: chex.Array
    banks: chex.Array

class BankHeistInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxBankHeist(JaxEnvironment[BankHeistState, BankHeistObservation, BankHeistInfo]):
    
    def __init__(self):
        super().__init__()
        self.frameskip = 1
        self.frame_stack_size = 4
        self.action_set = {NOOP, FIRE, RIGHT, LEFT, UP, DOWN}
        self.reward_funcs = None
    
    def reset(self) -> BankHeistState:
        # Minimal state initialization
        state = BankHeistState(
            level=jnp.array(0).astype(jnp.int32),
            difficulty_level=jnp.array(0).astype(jnp.int32),
            fuel=jnp.array(FUEL_CAPACITY).astype(jnp.float32),
            player=Entity(
                position=jnp.array([12, 78]).astype(jnp.int32),
                direction=jnp.array(4).astype(jnp.int32),
                visibility=jnp.array([1]).astype(jnp.int32)
            ),
            dynamite_position=jnp.array([-1, -1]).astype(jnp.int32),  # Inactive at [-1, -1]
            enemy_positions=init_banks_or_police(),
            bank_positions=init_banks_or_police(),
            speed=jnp.array(BASE_SPEED).astype(jnp.float32),
            fuel_consumption=jnp.array(1).astype(jnp.float32),
            reserve_speed=jnp.array(0.0).astype(jnp.float32),
            police_reserve_speed=jnp.array(0.0).astype(jnp.float32),
            money=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(STARTING_LIVES).astype(jnp.int32),
            fuel_refill=jnp.array(0).astype(jnp.int32),
            obs_stack=None,
            map_collision=CITY_COLLISION_MAPS[0],
            spawn_points=CITY_SPAWNS[0],
            bank_spawn_timers=jnp.array([1, 1, 1]).astype(jnp.int32),
            police_spawn_timers=jnp.array([-1, -1, -1]).astype(jnp.int32),
            dynamite_timer=jnp.array([-1]).astype(jnp.int32),
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),  # -1 means no pending spawn
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32),  # Bank indices for pending spawns
            game_paused=jnp.array(True).astype(jnp.bool_),
            bank_heists=jnp.array(0).astype(jnp.int32),
            random_key=jax.random.PRNGKey(42)  # Initialize with a fixed seed
        )
        obs = self._get_observation(state)
        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
        obs_stack = jax.tree.map(expand_and_copy, obs)
        state = state._replace(obs_stack=obs_stack)
        return  obs_stack, state

    
    @partial(jax.jit, static_argnums=(0,))
    def validate_input(self, state: BankHeistState, player: Entity, input: jnp.ndarray) -> Entity:
        """
        Confirm that the player is not trying to move into a wall.

        Returns:
            EntityPosition: Contains the new direction of the player after validating the input.
        """
        new_position = self.move(player, input)
        new_position = new_position._replace(direction=input)
        collision = self.check_background_collision(state, new_position)
        direction = jax.lax.cond(collision >= 255,
            lambda: player.direction,
            lambda: new_position.direction
        )
        return player._replace(direction=direction), collision >= 255

    @partial(jax.jit, static_argnums=(0,))
    def check_background_collision(self, state: BankHeistState, new_position: Entity) -> int:
        """
        Check for collisions with the background (walls, portals).

        Returns:
            int: The maximum collision value found(255: wall, 100: portal, 200: exit, 0: empty space).
        """
        new_coords = jnp.array([new_position.position[0], new_position.position[1]-1])
        new_position_bg: jnp.ndarray = jax.lax.dynamic_slice(operand=state.map_collision,
                          start_indices=new_coords, slice_sizes=COLLISION_BOX)
        max_value = jnp.max(new_position_bg)
        return max_value

    @partial(jax.jit, static_argnums=(0,))
    def portal_handler(self, entity: Entity, collision: int) -> Entity:
        """
        Handle portal collisions by moving the entity to the corresponding portal exit.

        Returns:
            Entity: The new position of the entity after handling the portal collision.
        """
        side = entity.position[0] <= 80
        side = side.astype(int)
        portal_collision = collision == 100
        new_position = jax.lax.cond(portal_collision,
            lambda: entity._replace(position=jnp.array([PORTAL_X[side], entity.position[1]])),
            lambda: entity
        )
        return new_position

    @partial(jax.jit, static_argnums=(0,))
    def check_bank_collision(self, player: Entity, banks: Entity) -> Tuple[chex.Array, chex.Array]:
        """
        Check if the player collides with any visible banks.

        Returns:
            Tuple of (collision_mask, bank_index) where collision_mask indicates which banks were hit
            and bank_index is the index of the first bank hit (-1 if none).
        """
        # Calculate distance between player and each bank
        player_pos = player.position
        bank_positions = banks.position
        
        # Check collision for each bank (simple distance-based collision)
        distances = jnp.linalg.norm(bank_positions - player_pos[None, :], axis=1)
        collision_distance = 8  # Collision threshold
        
        # Only consider visible banks
        visible_mask = banks.visibility > 0
        collision_mask = (distances < collision_distance) & visible_mask
        
        # Find first colliding bank index (-1 if none)
        bank_index = jnp.where(collision_mask, jnp.arange(len(collision_mask)), -1)
        first_bank_hit = jnp.max(bank_index)  # Get the first valid index or -1
        
        return collision_mask, first_bank_hit

    @partial(jax.jit, static_argnums=(0,))
    def check_police_collision(self, player: Entity, police: Entity) -> Tuple[chex.Array, chex.Array]:
        """
        Check if the player collides with any visible police cars.

        Args:
            player: The player entity with position
            police: The police entities with positions and visibility

        Returns:
            Tuple of (collision_mask, police_index) where collision_mask indicates which police cars were hit
            and police_index is the index of the first police car hit (-1 if none).
        """
        # Calculate distance between player and each police car
        player_pos = player.position
        police_positions = police.position
        
        # Check collision for each police car (simple distance-based collision)
        distances = jnp.linalg.norm(police_positions - player_pos[None, :], axis=1)
        collision_distance = 8  # Collision threshold
        
        # Only consider visible police cars
        visible_mask = police.visibility > 0
        collision_mask = (distances < collision_distance) & visible_mask
        
        # Find first colliding police car index (-1 if none)
        police_index = jnp.where(collision_mask, jnp.arange(len(collision_mask)), -1)
        first_police_hit = jnp.max(police_index)  # Get the first valid index or -1
        
        return collision_mask, first_police_hit

    @partial(jax.jit, static_argnums=(0,))
    def handle_bank_robbery(self, state: BankHeistState, bank_hit_index: chex.Array) -> BankHeistState:
        """
        Handle a bank robbery by hiding the bank and setting up delayed police spawn.

        Args:
            state: Current game state
            bank_hit_index: Index of the bank that was robbed

        Returns:
            BankHeistState: Updated state with bank hidden and police spawn scheduled
        """
        # Hide the robbed bank
        new_bank_visibility = state.bank_positions.visibility.at[bank_hit_index].set(0)
        new_banks = state.bank_positions._replace(visibility=new_bank_visibility)
        
        # Find an available pending spawn slot
        available_spawn_slots = state.pending_police_spawns < 0
        slot_index = jnp.where(available_spawn_slots, jnp.arange(len(available_spawn_slots)), len(available_spawn_slots))
        first_available_slot = jnp.min(slot_index)
        first_available_slot = jnp.where(first_available_slot >= len(available_spawn_slots), 0, first_available_slot)
        
        # Set up delayed spawn using constant
        new_pending_spawns = state.pending_police_spawns.at[first_available_slot].set(POLICE_SPAWN_DELAY)
        new_pending_bank_indices = state.pending_police_bank_indices.at[first_available_slot].set(bank_hit_index)

        # Increase bank heists count by 1
        new_bank_heists = state.bank_heists + 1

        # Increase score by bank robbery reward
        new_money = state.money + BANK_ROBBERY_REWARD

        return state._replace(
            bank_positions=new_banks,
            pending_police_spawns=new_pending_spawns,
            pending_police_bank_indices=new_pending_bank_indices,
            bank_heists=new_bank_heists,
            money=new_money
        )

    @partial(jax.jit, static_argnums=(0,))
    def lose_life(self, state: BankHeistState) -> BankHeistState:
        """
        Handle collision with police cars by reducing player lives and resetting player position.

        Args:
            state: Current game state
            police_hit_index: Index of the police car that was hit

        Returns:
            BankHeistState: Updated state with reduced lives and reset player position
        """
        # Reduce player lives by 1
        new_player_lives = state.player_lives - 1
        
        # Reset player to default position (same as level transition)
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player._replace(position=default_player_position)

        # Reset all police cars to spawn point (76, 132)
        reset_police_position = jnp.array([76, 132]).astype(jnp.int32)
        # Create array with reset position for all police cars
        reset_positions = jnp.tile(reset_police_position[None, :], (state.enemy_positions.position.shape[0], 1))
        # Only reset visible police cars, keep invisible ones at their current position
        visible_mask = state.enemy_positions.visibility > 0
        new_police_positions = jnp.where(visible_mask[:, None], reset_positions, state.enemy_positions.position)
        new_police = state.enemy_positions._replace(position=new_police_positions)
        
        return state._replace(
            player_lives=new_player_lives,
            player=new_player,
            enemy_positions=new_police,
            game_paused=jnp.array(True)
        )

    @partial(jax.jit, static_argnums=(0,))
    def spawn_police_car(self, state: BankHeistState, bank_index: chex.Array) -> BankHeistState:
        """
        Spawn a police car at the position of the specified bank index.

        Returns:
            BankHeistState: Updated state with a police car spawned.
        """
        # Find an available police slot (visibility == 0)
        available_slots = state.enemy_positions.visibility == 0
        
        # If no slots available, use the first slot
        slot_index = jnp.where(available_slots, jnp.arange(len(available_slots)), len(available_slots))
        first_available = jnp.min(slot_index)
        first_available = jnp.where(first_available >= len(available_slots), 0, first_available)
        
        # Get spawn position from bank index
        spawn_position = state.bank_positions.position[bank_index]
        
        # Update police positions
        new_positions = state.enemy_positions.position.at[first_available].set(spawn_position)
        new_directions = state.enemy_positions.direction.at[first_available].set(4)  # Default direction
        new_visibility = state.enemy_positions.visibility.at[first_available].set(1)  # Make visible
        
        new_police = state.enemy_positions._replace(
            position=new_positions,
            direction=new_directions, 
            visibility=new_visibility
        )
        
        return state._replace(enemy_positions=new_police)

    @partial(jax.jit, static_argnums=(0,))
    def process_pending_police_spawns(self, state: BankHeistState) -> BankHeistState:
        """
        Process all pending police spawns that are ready (timer == 0).

        Returns:
            BankHeistState: Updated state with police cars spawned and pending spawns cleared.
        """
        def process_single_spawn(i, current_state):
            # Check if this spawn slot is ready
            def spawn_at_bank_index(state_inner):
                bank_index = state_inner.pending_police_bank_indices[i]
                spawned_state = self.spawn_police_car(state_inner, bank_index)
                # Clear the pending spawn
                new_pending_spawns = spawned_state.pending_police_spawns.at[i].set(-1)
                new_pending_indices = spawned_state.pending_police_bank_indices.at[i].set(-1)
                return spawned_state._replace(
                    pending_police_spawns=new_pending_spawns,
                    pending_police_bank_indices=new_pending_indices
                )
            
            ready_to_spawn = current_state.pending_police_spawns[i] == 0
            return jax.lax.cond(ready_to_spawn, spawn_at_bank_index, lambda s: s, current_state)
        
        return jax.lax.fori_loop(0, len(state.pending_police_spawns), process_single_spawn, state)
    
    @partial(jax.jit, static_argnums=(0,))
    def place_dynamite(self, state: BankHeistState, player: Entity) -> BankHeistState:
        """
        Place dynamite 4 pixels behind the player if dynamite is currently inactive.

        Args:
            state: Current game state
            player: Current player entity with position and direction

        Returns:
            BankHeistState: Updated state with dynamite placed (if it was inactive)
        """
        # Check if dynamite is currently inactive (at [-1, -1])
        dynamite_inactive = jnp.all(state.dynamite_position == jnp.array([-1, -1]))
        
        def place_new_dynamite():
            # Calculate position 4 pixels behind the player based on their direction
            player_position = player.position
            player_direction = player.direction
            
            # Calculate offset based on direction (opposite to movement direction)
            offset_branches = [
                lambda: jnp.array([0, -5]),   # DOWN -> place 4 pixels UP (behind)
                lambda: jnp.array([0, 5]),    # UP -> place 4 pixels DOWN (behind)
                lambda: jnp.array([-5, 2]),   # RIGHT -> place 4 pixels LEFT (behind)
                lambda: jnp.array([5, 2]),    # LEFT -> place 4 pixels RIGHT (behind)
                lambda: jnp.array([0, 0]),    # NOOP/FIRE -> place at current position
            ]
            offset = jax.lax.switch(player_direction, offset_branches)
            
            # Place dynamite 4 pixels behind the player
            new_dynamite_position = player_position + offset
            new_dynamite_timer = jnp.array([DYNAMITE_EXPLOSION_DELAY]).astype(jnp.int32)  # 90 frames until explosion
            return state._replace(
                dynamite_position=new_dynamite_position,
                dynamite_timer=new_dynamite_timer,
                fuel=state.fuel - DYNAMITE_COST
            )
        # Reduce Fuel if dynamite was placed
        # Only place dynamite if it's currently inactive
        return jax.lax.cond(dynamite_inactive, place_new_dynamite, lambda: state)

    @partial(jax.jit, static_argnums=(0,))
    def kill_police_in_explosion_range(self, state: BankHeistState) -> BankHeistState:
        """
        Kill police cars within a 5x5 range of the dynamite explosion.

        Args:
            state: Current game state

        Returns:
            BankHeistState: Updated state with police cars killed if within explosion range
        """
        dynamite_pos = state.dynamite_position
        police_killed_count = jnp.array(0)
        
        def check_and_kill_police(i, carry):
            current_state, killed_count = carry
            police_pos = current_state.enemy_positions.position[i]
            police_visible = current_state.enemy_positions.visibility[i] > 0
            
            # Calculate distance between dynamite and police car
            distance = jnp.linalg.norm(dynamite_pos - police_pos)
            
            # Kill police car if it's visible and within 5x5 range (distance <= ~3.5 pixels)
            # Using 2.5 as threshold for a 5x5 area (half the diagonal of 5x5 square)
            within_range = distance <= 5
            should_kill = police_visible & within_range
            
            # Set visibility to 0 (kill) if within range
            new_visibility = current_state.enemy_positions.visibility.at[i].set(
                jnp.where(should_kill, 0, current_state.enemy_positions.visibility[i])
            )
            
            new_police = current_state.enemy_positions._replace(visibility=new_visibility)
            updated_state = current_state._replace(enemy_positions=new_police)
            
            # Increment killed count if a police car was killed
            new_killed_count = killed_count + jnp.where(should_kill, 1, 0)
            
            return (updated_state, new_killed_count)
        
        # Process all police cars and count kills
        final_state, total_killed = jax.lax.fori_loop(
            0, len(state.enemy_positions.visibility), 
            check_and_kill_police, 
            (state, police_killed_count)
        )
        
        # Add score for killed police cars
        new_money = final_state.money + (total_killed * POLICE_KILL_REWARD)

        return final_state._replace(money=new_money)
    
    @partial(jax.jit, static_argnums=(0,))
    def explode_dynamite(self, state: BankHeistState) -> BankHeistState:
        """
        Handle dynamite explosion - kill police cars within 5x5 range and make dynamite inactive.

        Args:
            state: Current game state

        Returns:
            BankHeistState: Updated state with police cars killed and dynamite deactivated
        """
        # Kill police cars within 5x5 range of dynamite explosion
        updated_state = self.kill_police_in_explosion_range(state)
        
        # Deactivate the dynamite by setting position to [-1, -1] and timer to -1
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        
        return updated_state._replace(
            dynamite_position=new_dynamite_position,
            dynamite_timer=new_dynamite_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: BankHeistState) -> BankHeistState:
        new_level = state.level+1
        new_difficulty_level = jnp.minimum(new_level // CITIES_PER_LEVEL, MAX_LEVEL-1)
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player._replace(position=default_player_position)
        empty_police = init_banks_or_police()
        empty_banks = init_banks_or_police()
        new_speed = BASE_SPEED * jnp.power(SPEED_INCREASE_PER_LEVEL, new_difficulty_level)
        new_fuel_consumption = jnp.power(FUEL_CONSUMPTION_INCREASE_PER_LEVEL, new_difficulty_level)
        new_fuel = jnp.maximum(state.fuel, jax.lax.dynamic_index_in_dim(REFILL_TABLE, state.bank_heists, axis=0, keepdims=False))
        new_fuel_refill=jnp.array(0).astype(jnp.int32)
        map_id = new_level % len(CITY_COLLISION_MAPS)
        new_map_collision = jax.lax.dynamic_index_in_dim(CITY_COLLISION_MAPS, map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(CITY_SPAWNS, map_id, axis=0, keepdims=False)
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)  # Inactive dynamite
        new_bank_spawn_timers = jnp.array([1,1,1]).astype(jnp.int32)
        new_police_spawn_timers = jnp.array([-1,-1,-1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        new_player_lives = jax.lax.cond(state.bank_heists >= 9, lambda: state.player_lives + 1, lambda: state.player_lives)
        return state._replace(
            level=new_level,
            difficulty_level=new_difficulty_level,
            player=new_player,
            player_lives=new_player_lives,
            enemy_positions=empty_police,
            bank_positions=empty_banks,
            speed=new_speed,
            fuel_consumption=new_fuel_consumption,
            fuel=new_fuel,
            fuel_refill=new_fuel_refill,
            map_collision=new_map_collision,
            spawn_points=new_spawn_points,
            dynamite_position=new_dynamite_position,
            bank_spawn_timers=new_bank_spawn_timers,
            police_spawn_timers=new_police_spawn_timers,
            dynamite_timer=new_dynamite_timer,
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32),
            bank_heists=jnp.array(0).astype(jnp.int32),
            random_key=jax.random.PRNGKey(new_level + 100)  # New random key for new level
        )

    @partial(jax.jit, static_argnums=(0,))
    def move(self, position: Entity, direction: int) -> Entity:
        """
        Move the player in the specified direction by the specified speed.

        Returns:
            EntityPosition: The new position of the player after moving.
        """
        new_position = position
        branches = [
            lambda: new_position._replace(position=jnp.array([new_position.position[0], new_position.position[1] + 1])),  # DOWN
            lambda: new_position._replace(position=jnp.array([new_position.position[0], new_position.position[1] - 1])),  # UP
            lambda: new_position._replace(position=jnp.array([new_position.position[0] + 1, new_position.position[1]])),  # RIGHT
            lambda: new_position._replace(position=jnp.array([new_position.position[0] - 1, new_position.position[1]])),  # LEFT
            lambda: new_position,  # NOOP
        ]
        new_position = jax.lax.switch(direction, branches)
        return new_position

    @partial(jax.jit, static_argnums=(0,))
    def check_valid_direction(self, state: BankHeistState, position: chex.Array, direction: int) -> bool:
        """
        Check if a direction is valid (no collision with walls).
        
        Returns:
            bool: True if the direction is valid, False otherwise.
        """
        # Create a temporary entity to test the movement
        temp_entity = Entity(position=position,direction=jnp.array(direction), visibility=jnp.array(1))
        new_position = self.move(temp_entity, direction)
        collision = self.check_background_collision(state, new_position)
        return collision < 255  # Valid if not hitting a wall

    @partial(jax.jit, static_argnums=(0,))
    def get_valid_directions(self, state: BankHeistState, position: chex.Array, current_direction: int) -> chex.Array:
        """
        Get all valid directions from the current position, excluding reverse direction.
        
        Returns:
            chex.Array: Array of valid directions (0-3), with -1 for invalid slots.
        """
        # All possible directions (excluding NOOP and FIRE)
        all_directions = jnp.array([DOWN, UP, RIGHT, LEFT])
        
        # Calculate reverse direction
        reverse_direction = jax.lax.switch(current_direction, [
            lambda: UP,    # If going DOWN, reverse is UP
            lambda: DOWN,  # If going UP, reverse is DOWN  
            lambda: LEFT,  # If going RIGHT, reverse is LEFT
            lambda: RIGHT, # If going LEFT, reverse is RIGHT
            lambda: -1,    # If NOOP, no reverse
        ])
        
        # Check which directions are valid
        def check_direction(direction):
            is_reverse = direction == reverse_direction
            is_valid = self.check_valid_direction(state, position, direction)
            return is_valid & (~is_reverse)
        
        valid_mask = jax.vmap(check_direction)(all_directions)
        
        # Create array with valid directions, -1 for invalid
        valid_directions = jnp.where(valid_mask, all_directions, -1)
        return valid_directions

    @partial(jax.jit, static_argnums=(0,))
    def choose_police_direction(self, state: BankHeistState, police_position: chex.Array, current_direction: int, random_key: chex.PRNGKey) -> int:
        """
        Choose the next direction for a police car using simple AI biased towards the player.
        
        Returns:
            int: The chosen direction.
        """
        valid_directions = self.get_valid_directions(state, police_position, current_direction)
        player_position = state.player.position
        
        # Count valid directions
        valid_count = jnp.sum(valid_directions >= 0)
        
        # If no valid directions, continue in current direction (or stay put)
        def no_valid_directions():
            return current_direction
        
        # If only one valid direction, take it
        def one_valid_direction():
            # Find the first valid direction (should be the only one)
            valid_mask = valid_directions >= 0
            # Use where to get the first valid direction
            return jnp.where(valid_mask, valid_directions, 0).max()
        
        # If multiple valid directions, choose with bias towards player
        def multiple_valid_directions():
            # Calculate what the new position would be for each direction
            def get_new_position(direction):
                # Create temporary entity and move it
                temp_entity = Entity(position=police_position, direction=jnp.array(direction), visibility=jnp.array(1))
                moved_entity = self.move(temp_entity, direction)
                return moved_entity.position
            
            # Get new positions for all directions
            new_positions = jax.vmap(get_new_position)(jnp.array([DOWN, UP, RIGHT, LEFT]))
            
            # Calculate distances to player for each direction
            distances = jnp.linalg.norm(new_positions - player_position[None, :], axis=1)
            
            # Create bias weights: smaller distance = higher weight
            # Use negative distance so closer positions get higher values
            distance_bias = -distances
            
            # Normalize distance bias to prevent extreme values
            distance_bias = distance_bias - jnp.min(distance_bias)  # Make minimum 0
            max_bias = jnp.max(distance_bias)
            distance_bias = jnp.where(max_bias > 0, distance_bias / max_bias, 0.0)  # Normalize to 0-1
            
            # Create base weights: 1.0 for valid directions, 0.0 for invalid
            base_weights = jnp.where(valid_directions >= 0, 1.0, 0.0)
            
            # Combine random factor with distance bias using constants
            random_noise = jax.random.uniform(random_key, shape=(4,))
            combined_weights = base_weights * (
                POLICE_RANDOM_FACTOR * random_noise + 
                POLICE_BIAS_FACTOR * (distance_bias + 0.1)  # Add small constant to prevent zero weights
            )
            
            # Choose the direction with highest combined weight
            chosen_idx = jnp.argmax(combined_weights)
            return valid_directions[chosen_idx]
        
        return jax.lax.cond(
            valid_count == 0,
            no_valid_directions,
            lambda: jax.lax.cond(
                valid_count == 1,
                one_valid_direction,
                multiple_valid_directions
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def move_police_cars(self, state: BankHeistState, random_key: chex.PRNGKey) -> BankHeistState:
        """
        Move all visible police cars using simple AI.
        
        Returns:
            BankHeistState: Updated state with police cars moved.
        """
        def move_single_police(i, current_state):
            # Only move visible police cars
            def move_police_car(state_inner):
                police_position = state_inner.enemy_positions.position[i]
                current_direction = state_inner.enemy_positions.direction[i]
                
                # Generate random key for this police car
                police_key = jax.random.fold_in(random_key, i)
                
                # Choose new direction
                new_direction = self.choose_police_direction(state_inner, police_position, current_direction, police_key)
                
                # Move the police car
                temp_entity = Entity(
                    position=police_position,
                    direction=jnp.array(new_direction),
                    visibility=jnp.array(1)
                )
                moved_entity = self.move(temp_entity, new_direction)
                
                # Check for collisions and handle portals (but not walls or city transitions)
                collision = self.check_background_collision(state_inner, moved_entity)
                
                # Handle wall collisions (stop movement)
                moved_entity = jax.lax.cond(collision >= 255,
                    lambda: temp_entity,  # Don't move if hitting wall
                    lambda: moved_entity  # Allow movement
                )
                
                # Handle portal teleportation (same as player)
                moved_entity = self.portal_handler(moved_entity, collision)
                
                # Update police positions
                new_positions = state_inner.enemy_positions.position.at[i].set(moved_entity.position)
                new_directions = state_inner.enemy_positions.direction.at[i].set(new_direction)
                
                new_police = state_inner.enemy_positions._replace(
                    position=new_positions,
                    direction=new_directions
                )
                
                return state_inner._replace(enemy_positions=new_police)
            
            is_visible = current_state.enemy_positions.visibility[i] > 0
            return jax.lax.cond(is_visible, move_police_car, lambda s: s, current_state)
        
        return jax.lax.fori_loop(0, len(state.enemy_positions.visibility), move_single_police, state)

    @partial(jax.jit, static_argnums=(0,))
    def advance_random_key(self, state: BankHeistState) -> Tuple[chex.PRNGKey, BankHeistState]:
        """
        Split the random key and advance the state's random key for the next step.
        
        Returns:
            Tuple of (current_step_key, updated_state)
        """
        step_random_key, next_random_key = jax.random.split(state.random_key)
        updated_state = state._replace(random_key=next_random_key)
        return step_random_key, updated_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        # Get random key for this step and advance state's random key
        step_random_key, new_state = self.advance_random_key(state)

        full_speed = state.speed + state.reserve_speed
        state = state._replace(reserve_speed=jnp.mod(full_speed, 1.0))
        full_speed = full_speed.astype(jnp.int32)
        # Player step. Must be run first as this step unpauses the game if it is paused!
        new_state = jax.lax.fori_loop(0, full_speed, lambda i, s: self.player_step(s, action), state)
        police_speed = jax.lax.cond(
            new_state.game_paused,
            lambda: jnp.array(0).astype(jnp.float32),
            lambda: (state.speed * POLICE_SLOW_DOWN_FACTOR + state.police_reserve_speed)        )
        new_state = new_state._replace(police_reserve_speed=jnp.mod(police_speed, 1.0))
        police_speed = police_speed.astype(jnp.int32)
        new_state = jax.lax.fori_loop(0, police_speed, lambda i, s: self.move_police_cars(s, step_random_key), new_state)

        # Timer step
        new_state = jax.lax.cond(
            new_state.game_paused,
            lambda: new_state,
            lambda: self.timer_step(new_state,step_random_key)
        )

        # Fuel Consumption
        new_state = jax.lax.cond(
            new_state.game_paused,
            lambda: new_state,
            lambda: self.fuel_step(new_state)
        )

        return state.obs_stack, new_state, 0.0, 1, {}

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: BankHeistState, action: chex.Array) -> BankHeistState:
        """
        Handles player Input & movement.

        Returns:
            BankHeistState: The new state of the game after the player's action.
        """
        player_input = jnp.where(action == NOOP, state.player.direction, action)  # Convert NOOP to direction 4
        player_input = jnp.where(action == FIRE, state.player.direction, player_input)  # Ignore FIRE for movement
        current_player, invalid_input = self.validate_input(state, state.player, player_input)
        new_player = self.move(current_player, current_player.direction)
        collision = self.check_background_collision(state, new_player)
        new_player = jax.lax.cond(collision >= 255,
            lambda: current_player,
            lambda: new_player
        )
        new_player = self.portal_handler(new_player, collision)

        new_state = state._replace(
            player=new_player,
            game_paused=jnp.where(jnp.logical_and(jnp.logical_not(invalid_input), action != NOOP), jnp.array(False).astype(jnp.bool_), state.game_paused)
        )

        # Check for bank collisions and handle bank robberies (before any other state changes)
        bank_collision_mask, bank_hit_index = self.check_bank_collision(new_player, state.bank_positions)
        
        # Apply bank robbery logic if any bank was hit
        bank_hit = bank_hit_index >= 0
        new_state = jax.lax.cond(bank_hit, lambda: self.handle_bank_robbery(new_state, bank_hit_index), lambda: new_state)

        # Check for police collisions and handle player death
        police_collision_mask, police_hit_index = self.check_police_collision(new_player, state.enemy_positions)
        
        # Apply police collision logic if any police car was hit
        police_hit = police_hit_index >= 0
        new_state = jax.lax.cond(police_hit, lambda: self.lose_life(new_state), lambda: new_state)

        # Handle dynamite placement when FIRE action is pressed
        new_state = jax.lax.cond(action == FIRE, 
            lambda: self.place_dynamite(new_state, new_player),
            lambda: new_state
        )

        new_state = jax.lax.cond(collision == 200, lambda: self.map_transition(new_state), lambda: new_state)

        new_state = jax.lax.cond(
            jnp.logical_and(new_state.game_paused, jnp.logical_not(police_hit)),
            lambda: state,
            lambda: new_state
        )
        return new_state
    
    @partial(jax.jit, static_argnums=(0,))
    def fuel_step(self, state: BankHeistState) -> BankHeistState:
        """
        Handles fuel consumption for the player's vehicle.
        """
        new_fuel = jnp.maximum(state.fuel - state.fuel_consumption, 0)

        new_state = state._replace(fuel=new_fuel)
        new_state = jax.lax.cond(
            new_state.fuel <= 0,
            lambda: self.lose_life(new_state)._replace(fuel=REVIVAL_FUEL),
            lambda: new_state
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def timer_step(self, state: BankHeistState, step_random_key: chex.PRNGKey) -> BankHeistState:
        """
        Handles the countdown of timers for the spawning of police cars and banks as well as dynamite explosions.

        Returns:
            BankHeistState: The new state of the game after the timer step.
        """
        def spawn_bank(state: BankHeistState) -> BankHeistState:
            # Use the step random key for bank spawning
            new_bank_spawns = jax.random.randint(step_random_key, shape=(state.bank_positions.position.shape[0],), minval=0, maxval=state.spawn_points.shape[0])
            chosen_points = state.spawn_points[new_bank_spawns]
            mask = (state.bank_spawn_timers == 0)[:, None]  # shape (3, 1)
            new_bank_positions = jnp.where(mask, chosen_points, state.bank_positions.position)

            new_visibility = jnp.where(state.bank_spawn_timers == 0, jnp.array([1,1,1]), state.bank_positions.visibility)

            new_banks = state.bank_positions._replace(position=new_bank_positions, visibility=new_visibility)
            return state._replace(bank_positions=new_banks)

        new_bank_spawn_timers = jnp.where(state.bank_spawn_timers >= 0, state.bank_spawn_timers - 1, state.bank_spawn_timers)
        new_police_spawn_timers = jnp.where(state.police_spawn_timers >= 0, state.police_spawn_timers - 1, state.police_spawn_timers)
        new_dynamite_timer = jnp.where(state.dynamite_timer >= 0, state.dynamite_timer - 1, state.dynamite_timer)
        
        # Handle pending police spawns
        new_pending_police_spawns = jnp.where(state.pending_police_spawns >= 0, state.pending_police_spawns - 1, state.pending_police_spawns)

        new_state = state._replace(
            bank_spawn_timers=new_bank_spawn_timers,
            police_spawn_timers=new_police_spawn_timers,
            dynamite_timer=new_dynamite_timer,
            pending_police_spawns=new_pending_police_spawns
        )
        
        # Spawn banks when their timers reach 0
        spawn_bank_condition = jnp.any(new_bank_spawn_timers == 0)
        new_state = jax.lax.cond(spawn_bank_condition, lambda: spawn_bank(new_state), lambda: new_state)
        
        # Process delayed police spawns
        spawn_police_condition = jnp.any(new_pending_police_spawns == 0)
        new_state = jax.lax.cond(spawn_police_condition, lambda: self.process_pending_police_spawns(new_state), lambda: new_state)
        
        # Handle dynamite explosion when timer reaches 0 (check original timer before decrementing)
        dynamite_explode_condition = state.dynamite_timer[0] == 0
        new_state = jax.lax.cond(dynamite_explode_condition, lambda: self.explode_dynamite(new_state), lambda: new_state)
        
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BankHeistState) -> BankHeistObservation:
        return BankHeistObservation(
            player=state.player,
            lives=state.player_lives,
            score=state.money,
            enemies=state.enemy_positions,
            banks=state.bank_positions,
            )


def load_bankheist_sprites():
    cities = [aj.loadFrame(os.path.join(SPRITES_DIR, f"map_{i+1}.npy"), transpose=True) for i in range(8)]
    player_side = aj.loadFrame(os.path.join(SPRITES_DIR, "player_side.npy"), transpose=True)
    player_front = aj.loadFrame(os.path.join(SPRITES_DIR, "player_front.npy"), transpose=True)
    police_side = aj.loadFrame(os.path.join(SPRITES_DIR, "police_side.npy"), transpose=True)
    police_front = aj.loadFrame(os.path.join(SPRITES_DIR, "police_front.npy"), transpose=True)
    bank = aj.loadFrame(os.path.join(SPRITES_DIR, "bank.npy"), transpose=True)
    dynamite = aj.loadFrame(os.path.join(SPRITES_DIR, "dynamite_0.npy"), transpose=True)
    fuel_tank = aj.loadFrame(os.path.join(SPRITES_DIR, "fuel_tank.npy"), transpose=True)
    fuel_gauge = aj.loadFrame(os.path.join(SPRITES_DIR, "fuel_gauge.npy"), transpose=True)

    # Add padding to front sprites so they have same dimensions as side sprites
    player_front_padded = jnp.pad(player_front, ((1,1), (0,0), (0,0)), mode='constant')
    police_front_padded = jnp.pad(police_front, ((1,1), (0,0), (0,0)), mode='constant')

    CITY_SPRITES = jnp.stack([jnp.expand_dims(city, axis=0) for city in cities])
    PLAYER_SIDE_SPRITE = jnp.expand_dims(player_side, axis=0)
    PLAYER_FRONT_SPRITE = jnp.expand_dims(player_front_padded, axis=0)
    POLICE_SIDE_SPRITE = jnp.expand_dims(police_side, axis=0)
    POLICE_FRONT_SPRITE = jnp.expand_dims(police_front_padded, axis=0)
    BANK_SPRITE = jnp.expand_dims(bank, axis=0)
    DYNAMITE_SPRITE = jnp.expand_dims(dynamite, axis=0)
    FUEL_TANK_SPRITE = jnp.expand_dims(fuel_tank, axis=0)
    FUEL_GAUGE_SPRITE = jnp.expand_dims(fuel_gauge, axis=0)

    DIGIT_SPRITES = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, os.path.join(SPRITES_DIR, "score_{}.npy")),
        num_chars=10,
    )

    return (PLAYER_SIDE_SPRITE, PLAYER_FRONT_SPRITE, POLICE_SIDE_SPRITE, POLICE_FRONT_SPRITE, BANK_SPRITE,DYNAMITE_SPRITE, CITY_SPRITES, FUEL_TANK_SPRITE, FUEL_GAUGE_SPRITE, DIGIT_SPRITES)

class Renderer_AtraBankisHeist:
    def __init__(self):
        (
            self.SPRITE_PLAYER_SIDE,
            self.SPRITE_PLAYER_FRONT,
            self.SPRITE_POLICE_SIDE,
            self.SPRITE_POLICE_FRONT,
            self.SPRITE_BANK,
            self.SPRITE_DYNAMITE,
            self.SPRITES_CITY,
            self.SPRITE_FUEL_TANK,
            self.SPRITE_FUEL_GAUGE,
            self.DIGIT_SPRITES
        ) = load_bankheist_sprites() 

    @partial(jax.jit, static_argnums=(0,))
    def fuel_tank_filler(self, sprite: chex.Array, state: BankHeistState) -> jax.Array:
        """
        Colors the bottom part of the fuel tank sprite red, based on current fuel level.

        Returns: Jax Array of same shape as input sprite
        """
        # Get the number of channels
        num_channels = sprite.shape[-1]
        
        # Extract channels
        r = sprite[..., 0]
        g = sprite[..., 1]
        b = sprite[..., 2]
        
        # Handle alpha channel if present
        has_alpha = num_channels >= 4
        alpha = jnp.ones_like(r) if not has_alpha else sprite[..., 3]
        # Calculate the fill level based on fuel
        level = state.fuel / FUEL_CAPACITY
        pixel_level = jnp.ceil(level * TANK_HEIGHT)
        # Create a mask for the pixels to be filled
        fill_mask = jnp.arange(TANK_HEIGHT) > TANK_HEIGHT-pixel_level

        final_r = jnp.where(fill_mask, 167, r)
        final_g = jnp.where(fill_mask, 26, g)
        final_b = jnp.where(fill_mask, 26, b)

        # Stack channels back together with alpha if it exists
        if has_alpha:
            return jnp.stack([final_r, final_g, final_b, alpha], axis=-1).astype(jnp.uint8)
        else:
            return jnp.stack([final_r, final_g, final_b], axis=-1).astype(jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render_lives(self, raster, state, life_frame):
        def body_fun(i, rast):
            return jax.lax.cond(
                i < state.player_lives,
                lambda r: aj.render_at(r, LIFE_POSITIONS[i][0], LIFE_POSITIONS[i][1], life_frame),
                lambda r: r,
                rast
            )
        raster = jax.lax.fori_loop(0, len(LIFE_POSITIONS), body_fun, raster)
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)

        ### Render City
        frame_city = aj.get_sprite_frame(self.SPRITES_CITY[state.level % self.SPRITES_CITY.shape[0]], 0)
        raster = aj.render_at(raster, 0, 0, frame_city)

        ### Render Player
        branches = [
            lambda: aj.get_sprite_frame(self.SPRITE_PLAYER_FRONT, 0),  # DOWN
            lambda: aj.get_sprite_frame(self.SPRITE_PLAYER_FRONT, 0),  # UP
            lambda: aj.get_sprite_frame(self.SPRITE_PLAYER_SIDE, 0),   # RIGHT
            lambda: jnp.flip(aj.get_sprite_frame(self.SPRITE_PLAYER_SIDE, 0), axis=0),   # LEFT, Frame is Mirrored
        ]
        # Make no Direction equal to right for rendering
        player_direction = jax.lax.cond(
            state.player.direction == 4,
            lambda: 2,
            lambda: state.player.direction
        )
        player_frame = jax.lax.switch(player_direction, branches)
        raster = aj.render_at(raster, state.player.position[0], state.player.position[1], player_frame)

        ### Render Fuel Tank
        fuel_tank_frame = aj.get_sprite_frame(self.SPRITE_FUEL_TANK, 0)
        # color in the bottom of the fuel tank in red
        fuel_tank_frame = self.fuel_tank_filler(fuel_tank_frame, state)
        raster = aj.render_at(raster, FUEL_TANK_POSITION[0], FUEL_TANK_POSITION[1], fuel_tank_frame)
        # Render the Fuel gauge
        fuel_gauge_position = jax.lax.dynamic_index_in_dim(TANK_LEVELS, state.bank_heists, axis=0, keepdims=False)
        fuel_gauge_frame = aj.get_sprite_frame(self.SPRITE_FUEL_GAUGE, 0)
        raster = jax.lax.cond(fuel_gauge_position > 0,
                              lambda: aj.render_at(raster, FUEL_TANK_POSITION[0]+8, FUEL_TANK_POSITION[1]+(TANK_HEIGHT - fuel_gauge_position), fuel_gauge_frame),
                              lambda: raster,
                              )

        ### Render Player Lives
        life_frame = aj.get_sprite_frame(self.SPRITE_PLAYER_SIDE, 0)
        raster = self.render_lives(raster, state, life_frame)

        ### Render Banks
        bank_frame = aj.get_sprite_frame(self.SPRITE_BANK, 0)
        for i in range(state.bank_positions.position.shape[0]):
            raster = jax.lax.cond(
                state.bank_positions.visibility[i] != 0,
                lambda r: aj.render_at(r, state.bank_positions.position[i, 0], state.bank_positions.position[i, 1], bank_frame),
                lambda r: r,
                raster
            )

        ### Render Police Cars
        police_branches = [
            lambda: aj.get_sprite_frame(self.SPRITE_POLICE_FRONT, 0),  # DOWN
            lambda: aj.get_sprite_frame(self.SPRITE_POLICE_FRONT, 0),  # UP
            lambda: jnp.flip(aj.get_sprite_frame(self.SPRITE_POLICE_SIDE, 0), axis=0),   # RIGHT
            lambda: aj.get_sprite_frame(self.SPRITE_POLICE_SIDE, 0),   # LEFT, Frame is Mirrored
        ]
        
        for i in range(state.enemy_positions.position.shape[0]):
            def render_police(raster_input):
                # Get police direction, default to right if direction is 4 (NOOP)
                police_direction = jax.lax.cond(
                    state.enemy_positions.direction[i] == 4,
                    lambda: 2,  # Default to RIGHT
                    lambda: state.enemy_positions.direction[i]
                )
                police_frame = jax.lax.switch(police_direction, police_branches)
                return aj.render_at(raster_input, state.enemy_positions.position[i, 0], state.enemy_positions.position[i, 1], police_frame)
            
            raster = jax.lax.cond(
                state.enemy_positions.visibility[i] != 0,
                render_police,
                lambda r: r,
                raster
            )

        ### Render Dynamite (only when active)
        def render_dynamite(raster_input):
            dynamite_frame = aj.get_sprite_frame(self.SPRITE_DYNAMITE, 0)
            return aj.render_at(raster_input, state.dynamite_position[0], state.dynamite_position[1], dynamite_frame)
        
        # Check if dynamite is active (not at [-1, -1])
        dynamite_active = ~jnp.all(state.dynamite_position == jnp.array([-1, -1]))
        raster = jax.lax.cond(dynamite_active, render_dynamite, lambda r: r, raster)

        ### Render Player Lives (top right corner)
        # Use the right-facing player sprite for lives display
        life_sprite = aj.get_sprite_frame(self.SPRITE_PLAYER_SIDE, 0)
        life_sprite_height, life_sprite_width = life_sprite.shape[:2]
        
        # Position lives in top right corner with some padding
        lives_start_x = WIDTH - (4 * (life_sprite_width + 2) ) - 12  # 4 lives with 5 pixel padding from right edge
        lives_y = 12  # 12 pixels from top
        
        # Render each life sprite
        def render_single_life(i, current_raster):
            def render_life_sprite(raster_input):
                life_x = lives_start_x + (i * (life_sprite_width + 2))
                return aj.render_at(raster_input, life_x, lives_y, life_sprite)
            
            # Only render if player has this many lives remaining
            has_life = i < state.player_lives
            return jax.lax.cond(has_life, render_life_sprite, lambda r: r, current_raster)
        
        # Render up to 4 life sprites using fori_loop
        raster = jax.lax.fori_loop(0, 4, render_single_life, raster)

        score_digits = aj.int_to_digits(state.money, max_digits=4)
        raster = aj.render_label_selective(
            raster, 90, 179, score_digits, self.DIGIT_SPRITES, 0, len(score_digits), spacing=12
        )

        return raster

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Assault Game")
    clock = pygame.time.Clock()

    game = JaxBankHeist()

    # Create the JAX renderer
    renderer = Renderer_AtraBankisHeist()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    while running:
        game_over = jnp.logical_or(
            jnp.less(curr_state.player_lives, 0),  # Player has 0 or fewer lives
            jnp.greater_equal(curr_state.money, 999999)  # Score reached 999999
        )

        if game_over:
            print(f"Game Over! Final Score: {curr_state.money}, Lives: {curr_state.player_lives}")
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                    event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()