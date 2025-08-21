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

EMPTY_SPACE_ID = 0
WALLS_ID = 1

WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

COLLISION_BOX = (8, 8)
PORTAL_X = jnp.array([12, 140])

# Police spawn delay (120 frames = 2 seconds at 60 FPS)
POLICE_SPAWN_DELAY = 120

# Police AI bias factors
POLICE_RANDOM_FACTOR = 0.7  # 70% random movement
POLICE_BIAS_FACTOR = 0.3    # 30% bias towards player

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
    player: Entity
    dynamite_position: chex.Array
    enemy_positions: Entity
    bank_positions: Entity
    speed: chex.Array
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
            fuel=jnp.array(90).astype(jnp.int32),
            player=Entity(
                position=jnp.array([12, 78]).astype(jnp.int32),
                direction=jnp.array(4).astype(jnp.int32),
                visibility=jnp.array([1]).astype(jnp.int32)
            ),
            dynamite_position=jnp.array([]).astype(jnp.int32),
            enemy_positions=init_banks_or_police(),
            bank_positions=init_banks_or_police(),
            speed=jnp.array(1).astype(jnp.int32),
            money=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(4).astype(jnp.int32),
            fuel_refill=jnp.array(0).astype(jnp.int32),
            obs_stack=None,
            map_collision=CITY_COLLISION_MAPS[0],
            spawn_points=CITY_SPAWNS[0],
            bank_spawn_timers=jnp.array([1, 1, 1]).astype(jnp.int32),
            police_spawn_timers=jnp.array([-1, -1, -1]).astype(jnp.int32),
            dynamite_timer=jnp.array([-1]).astype(jnp.int32),
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),  # -1 means no pending spawn
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32)  # Bank indices for pending spawns
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
        new_position = self.move(player, input, state.speed)
        new_position = new_position._replace(direction=input)
        collision = self.check_background_collision(state, new_position)
        direction = jax.lax.cond(collision >= 255,
            lambda: player.direction,
            lambda: new_position.direction
        )
        return player._replace(direction=direction)

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
    def portal_handler(self, car: Entity, collision: int) -> Entity:
        """
        Handle portal collisions by moving the player to the corresponding portal exit.

        Returns:
            EntityPosition: The new position of the player after handling the portal collision.
        """
        side = car.position[0] <= 80
        side = side.astype(int)
        portal_collision = collision == 100
        new_position = jax.lax.cond(portal_collision,
            lambda: car._replace(position=jnp.array([PORTAL_X[side], car.position[1]])),
            lambda: car
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
        
        return state._replace(
            bank_positions=new_banks,
            pending_police_spawns=new_pending_spawns,
            pending_police_bank_indices=new_pending_bank_indices
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
    def map_transition(self, state: BankHeistState) -> BankHeistState:

        new_level = state.level+1
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player._replace(position=default_player_position)
        empty_police = init_banks_or_police()
        empty_banks = init_banks_or_police()
        new_speed = state.speed * 1
        new_fuel = state.fuel_refill
        new_fuel_refill=jnp.array(0).astype(jnp.int32)
        map_id = new_level % len(CITY_COLLISION_MAPS)
        new_map_collision = jax.lax.dynamic_index_in_dim(CITY_COLLISION_MAPS, map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(CITY_SPAWNS, map_id, axis=0, keepdims=False)
        new_dynamite_position = jnp.array([]).astype(jnp.int32)
        new_bank_spawn_timers = jnp.array([1,1,1]).astype(jnp.int32)
        new_police_spawn_timers = jnp.array([-1,-1,-1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        return state._replace(
            level=new_level,
            player=new_player,
            enemy_positions=empty_police,
            bank_positions=empty_banks,
            speed=new_speed,
            fuel=new_fuel,
            fuel_refill=new_fuel_refill,
            map_collision=new_map_collision,
            spawn_points=new_spawn_points,
            dynamite_position=new_dynamite_position,
            bank_spawn_timers=new_bank_spawn_timers,
            police_spawn_timers=new_police_spawn_timers,
            dynamite_timer=new_dynamite_timer,
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def move(self, position: Entity, direction: int, speed: int) -> Entity:
        """
        Move the player in the specified direction by the specified speed.

        Returns:
            EntityPosition: The new position of the player after moving.
        """
        new_position = position
        branches = [
            lambda: new_position._replace(position=jnp.array([new_position.position[0], new_position.position[1] + speed])),  # DOWN
            lambda: new_position._replace(position=jnp.array([new_position.position[0], new_position.position[1] - speed])),  # UP
            lambda: new_position._replace(position=jnp.array([new_position.position[0] + speed, new_position.position[1]])),  # RIGHT
            lambda: new_position._replace(position=jnp.array([new_position.position[0] - speed, new_position.position[1]])),  # LEFT
            lambda: new_position,  # NOOP
        ]
        return jax.lax.switch(direction, branches)

    @partial(jax.jit, static_argnums=(0,))
    def check_valid_direction(self, state: BankHeistState, position: chex.Array, direction: int) -> bool:
        """
        Check if a direction is valid (no collision with walls).
        
        Returns:
            bool: True if the direction is valid, False otherwise.
        """
        # Create a temporary entity to test the movement
        temp_entity = Entity(position=position, direction=jnp.array(direction), visibility=jnp.array(1))
        new_position = self.move(temp_entity, direction, state.speed)
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
                moved_entity = self.move(temp_entity, direction, state.speed)
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
                moved_entity = self.move(temp_entity, new_direction, state_inner.speed)
                
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
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        # Generate random key for this step
        step_key = jax.random.PRNGKey(state.level + jnp.sum(state.player.position))
        
        # Player step
        new_state = self.player_step(state, action)
        
        # Police AI movement step
        new_state = self.move_police_cars(new_state, step_key)
        
        # Timer step
        new_state = self.timer_step(new_state)
        return state.obs_stack, new_state, 0.0, 1, {}

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: BankHeistState, action: chex.Array) -> BankHeistState:
        """
        Handles player Input & movement.

        Returns:
            BankHeistState: The new state of the game after the player's action.
        """
        player_input = jnp.where(action == NOOP, state.player.direction, action)  # Convert NOOP to direction 4
        current_player = self.validate_input(state, state.player, player_input)
        new_player = self.move(current_player, current_player.direction, state.speed)
        collision = self.check_background_collision(state, new_player)
        new_player = jax.lax.cond(collision >= 255,
            lambda: current_player,
            lambda: new_player
        )
        new_player = self.portal_handler(new_player, collision)

        new_state = state._replace(
            player=new_player,
            )

        # Check for bank collisions and handle bank robberies
        bank_collision_mask, bank_hit_index = self.check_bank_collision(new_player, state.bank_positions)
        
        # Apply bank robbery logic if any bank was hit
        bank_hit = bank_hit_index >= 0
        new_state = jax.lax.cond(bank_hit, lambda: self.handle_bank_robbery(new_state, bank_hit_index), lambda: new_state)

        new_state = jax.lax.cond(collision == 200, lambda: self.map_transition(new_state), lambda: new_state)
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def timer_step(self, state: BankHeistState) -> BankHeistState:
        """
        Handles the countdown of timers for the spawning of police cars and banks as well as dynamite explosions.

        Returns:
            BankHeistState: The new state of the game after the timer step.
        """
        def spawn_bank(state: BankHeistState) -> BankHeistState:
            key = jax.random.PRNGKey(0)  # Use a fixed key for reproducibility
            new_bank_spawns = jax.random.randint(key, shape=(state.bank_positions.position.shape[0],), minval=0, maxval=state.spawn_points.shape[0])
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
        
        #new_state = jnp.where(new_police_spawn_timers == 0, self.spawn_police(new_state), new_state)
        #new_state = jnp.where(new_dynamite_timer == 0, self.explode_dynamite(new_state), new_state)
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

    # Add padding to front sprites so they have same dimensions as side sprites
    player_front_padded = jnp.pad(player_front, ((1,1), (0,0), (0,0)), mode='constant')
    police_front_padded = jnp.pad(police_front, ((1,1), (0,0), (0,0)), mode='constant')

    CITY_SPRITES = jnp.stack([jnp.expand_dims(city, axis=0) for city in cities])
    PLAYER_SIDE_SPRITE = jnp.expand_dims(player_side, axis=0)
    PLAYER_FRONT_SPRITE = jnp.expand_dims(player_front_padded, axis=0)
    POLICE_SIDE_SPRITE = jnp.expand_dims(police_side, axis=0)
    POLICE_FRONT_SPRITE = jnp.expand_dims(police_front_padded, axis=0)
    BANK_SPRITE = jnp.expand_dims(bank, axis=0)

    return (PLAYER_SIDE_SPRITE, PLAYER_FRONT_SPRITE, POLICE_SIDE_SPRITE, POLICE_FRONT_SPRITE, BANK_SPRITE, CITY_SPRITES)

class Renderer_AtraBankisHeist:
    def __init__(self):
        (
            self.SPRITE_PLAYER_SIDE,
            self.SPRITE_PLAYER_FRONT,
            self.SPRITE_POLICE_SIDE,
            self.SPRITE_POLICE_FRONT,
            self.SPRITE_BANK,
            self.SPRITES_CITY,
        ) = load_bankheist_sprites() 

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