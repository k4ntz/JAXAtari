import os
from functools import partial
from typing import NamedTuple, Tuple, Optional
import jax
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct
import jaxatari.spaces as spaces
from jaxatari.environment import JAXAtariAction as Action

# Use the new rendering utilities
from jaxatari.rendering import jax_rendering_utils as render_utils
from jax.scipy.ndimage import map_coordinates
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import AutoDerivedConstants


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

def load_city_collision_map(file_name: str, sprites_dir: str) -> chex.Array:
    """
    Loads the city collision map from the sprites directory.
    """
    map = jnp.load(os.path.join(sprites_dir, file_name))
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




def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for BankHeist.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # Define file lists for groups
    city_files = [f"map_{i+1}.npy" for i in range(8)]
    
    # Define procedural sprites
    # This color is used to fill the fuel tank
    fuel_fill_color = (167, 26, 26) 
    fuel_color_rgba = jnp.array(fuel_fill_color + (255,), dtype=jnp.uint8).reshape(1, 1, 4)
    config = (
        # Backgrounds (loaded as a group)
        # Note: The 'background' type is not used here, as the city map is the primary background.
        # We will treat 'cities' as our base background sprites.
        {'name': 'cities', 'type': 'group', 'files': city_files},
        
        # Player (loaded as single sprites for manual padding)
        {'name': 'player_side', 'type': 'single', 'file': 'player_side.npy'},
        {'name': 'player_front', 'type': 'single', 'file': 'player_front.npy'},
        
        # Police (loaded as single sprites for manual padding)
        {'name': 'police_side', 'type': 'single', 'file': 'police_side.npy'},
        {'name': 'police_front', 'type': 'single', 'file': 'police_front.npy'},
        
        # Other objects
        {'name': 'bank', 'type': 'single', 'file': 'bank.npy'},
        {'name': 'dynamite', 'type': 'single', 'file': 'dynamite_0.npy'},
        # This file is (11, 25, 4) on disk. It should be rendered as (11, 25).
        # We remove transpose=True.
        {'name': 'fuel_tank', 'type': 'single', 'file': 'fuel_tank.npy'}, 
        # This file is (3, 1, 4) on disk but should be rendered as (1, 3).
        {'name': 'fuel_gauge', 'type': 'single', 'file': 'fuel_gauge.npy', 'transpose': True},
        
        # UI
        {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        
        # Procedural
        {'name': 'fuel_color', 'type': 'procedural', 'data': fuel_color_rgba},
    )
    
    return config

class BankHeistConstants(AutoDerivedConstants):
    # Basic dimensions
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    
    # Internal direction codes (not Action constants)
    # These are used for internal direction representation: 0=DOWN, 1=UP, 2=RIGHT, 3=LEFT, 4=NOOP
    DIR_DOWN: int = struct.field(pytree_node=False, default=0)
    DIR_UP: int = struct.field(pytree_node=False, default=1)
    DIR_RIGHT: int = struct.field(pytree_node=False, default=2)
    DIR_LEFT: int = struct.field(pytree_node=False, default=3)
    DIR_NOOP: int = struct.field(pytree_node=False, default=4)
    
    # Fuel constants
    FUEL_CAPACITY: float = struct.field(pytree_node=False, default=10200.0)
    TANK_HEIGHT: float = struct.field(pytree_node=False, default=25.0)
    TANK_LEVELS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 1, 2, 4, 6, 9, 12, 16, 18, 25]))
    DYNAMITE_DELAY: int = struct.field(pytree_node=False, default=60)
    DYNAMITE_EXPLOSION_DELAY: int = struct.field(pytree_node=False, default=30)
    
    # Derived constants (dynamic calculation based on static fields)
    REFILL_TABLE: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    DYNAMITE_COST: Optional[float] = struct.field(pytree_node=False, default=None)
    REVIVAL_FUEL: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    
    # Speed and level constants
    BASE_SPEED: float = struct.field(pytree_node=False, default=0.5)
    SPEED_INCREASE_PER_LEVEL: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1.1))
    FUEL_CONSUMPTION_INCREASE_PER_LEVEL: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1.15))
    CITIES_PER_LEVEL: int = struct.field(pytree_node=False, default=4)
    MAX_LEVEL: int = struct.field(pytree_node=False, default=8)
    
    # Lives constants
    MAX_LIVES: int = struct.field(pytree_node=False, default=6)
    STARTING_LIVES: int = struct.field(pytree_node=False, default=4)
    FIRST_LIFE_POSITION: Tuple[int, int] = struct.field(pytree_node=False, default=(90, 27))  # (WIDTH-70, 27)
    LIFE_OFFSET: Tuple[int, int] = struct.field(pytree_node=False, default=(16, 12))
    
    # Derived constant (dynamic calculation based on static fields)
    LIFE_POSITIONS: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    
    # Reward constants
    BASE_BANK_ROBBERY_REWARD: int = struct.field(pytree_node=False, default=10)
    POLICE_KILL_REWARD: Tuple[int, int, int] = struct.field(pytree_node=False, default=(50, 30, 10))
    CITY_REWARD: Tuple[int, int, int, int] = struct.field(pytree_node=False, default=(93, 186, 279, 372))
    BONUS_REWARD: int = struct.field(pytree_node=False, default=372)
    
    # Position constants
    FUEL_TANK_POSITION: Tuple[int, int] = struct.field(pytree_node=False, default=(42, 12))
    COLLISION_BOX: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 8))
    PORTAL_X: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 138]))
    
    # Window dimensions (derived constants)
    # Derived constants (dynamic calculation based on static fields)
    WINDOW_WIDTH: Optional[int] = struct.field(pytree_node=False, default=None)
    WINDOW_HEIGHT: Optional[int] = struct.field(pytree_node=False, default=None)
    
    # Police constants
    POLICE_SLOW_DOWN_FACTOR: float = struct.field(pytree_node=False, default=0.9)
    BANK_RESPAWN_TIME: int = struct.field(pytree_node=False, default=300)
    POLICE_SPAWN_DELAY: int = struct.field(pytree_node=False, default=120)
    POLICE_RANDOM_FACTOR: float = struct.field(pytree_node=False, default=0.7)
    POLICE_BIAS_FACTOR: float = struct.field(pytree_node=False, default=0.4)
    
    # Path constants
    MODULE_DIR: str = struct.field(pytree_node=False, default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    SPRITES_DIR: str = struct.field(pytree_node=False, default_factory=lambda: os.path.join(render_utils.get_base_sprite_dir(), "bankheist"))
    
    # Asset config baked into constants
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)
    
    def compute_derived(self):
        """Compute derived constants based on static fields."""
        return {
            'REFILL_TABLE': self.TANK_LEVELS / self.TANK_HEIGHT * self.FUEL_CAPACITY,
            'DYNAMITE_COST': self.FUEL_CAPACITY * 0.02,
            'REVIVAL_FUEL': jnp.array(0.2 * self.FUEL_CAPACITY).astype(jnp.float32),
            'LIFE_POSITIONS': jnp.array([
                (self.FIRST_LIFE_POSITION[0] + i%3 * self.LIFE_OFFSET[0], 
                 self.FIRST_LIFE_POSITION[1] - (i//3) * self.LIFE_OFFSET[1]) 
                for i in range(6)
            ]),
            'WINDOW_WIDTH': self.WIDTH * 3,
            'WINDOW_HEIGHT': self.HEIGHT * 3,
        }

class Entity(struct.PyTreeNode):
    position: jnp.ndarray
    direction: jnp.ndarray
    visibility: jnp.ndarray

class FlatEntity(struct.PyTreeNode):
    x: jnp.ndarray
    y: jnp.ndarray
    direction: jnp.ndarray
    visibility: jnp.ndarray

def flat_entity(entity: Entity):
    return FlatEntity(entity.position[0], entity.position[1], entity.direction, entity.visibility)

class BankHeistState(struct.PyTreeNode):
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
    time: chex.Array

#TODO: Add Background collision Map, Fuel, Fuel Refill and others
class BankHeistObservation(struct.PyTreeNode):
    player: ObjectObservation
    enemies: ObjectObservation
    banks: ObjectObservation
    dynamite: ObjectObservation
    fuel: jnp.ndarray
    fuel_refill: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray

class BankHeistInfo(struct.PyTreeNode):
    time: jnp.ndarray

class JaxBankHeist(JaxEnvironment[BankHeistState, BankHeistObservation, BankHeistInfo, BankHeistConstants]):
    # Minimal ALE action set for Bank Heist
    ACTION_SET: jnp.ndarray = jnp.array(
        [
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
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: BankHeistConstants = None):
        consts = consts or BankHeistConstants()
        super().__init__(consts)
        self.consts = consts
        # Compute city collision maps and spawn points
        self.city_collision_maps = jnp.array([
            load_city_collision_map(f"map_{i+1}_collision.npy", self.consts.SPRITES_DIR) 
            for i in range(8)
        ])
        self.city_spawns = get_spawn_points(self.city_collision_maps)
        # Use the new renderer class
        self.renderer = BankHeistRenderer(self.consts)
    
    def action_space(self) -> spaces.Discrete:
        """
        Returns the action space of the environment.
        """
        return spaces.Discrete(len(self.ACTION_SET))
    
    def observation_space(self) -> spaces.Dict:
        """
        Returns the observation space of the environment.
        """
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "enemies": spaces.get_object_space(n=3, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "banks": spaces.get_object_space(n=3, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "dynamite": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "fuel": spaces.Box(low=0, high=self.consts.FUEL_CAPACITY, shape=(), dtype=jnp.float32),
            "fuel_refill": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
        })
    
    def image_space(self) -> spaces.Box:
        """Returns the image space for Freeway.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )
    
    def obs_to_flat_array(self, obs: BankHeistObservation) -> jnp.ndarray:
        """Convert observation to a flat array."""
        player_array = jnp.concatenate([obs.player.x.reshape(-1), obs.player.y.reshape(-1), obs.player.direction.reshape(-1), obs.player.visibility.reshape(-1)])
        banks_flat = obs.banks.reshape(-1)
        police_flat = obs.enemies.reshape(-1)
        flat_array = jnp.concatenate([player_array, obs.lives.reshape(-1), obs.score.reshape(-1), banks_flat, police_flat]) 
        return flat_array
    

    def reset(self, key: chex.PRNGKey) -> BankHeistState:
        # Minimal state initialization
        state = BankHeistState(
            level=jnp.array(0).astype(jnp.int32),
            difficulty_level=jnp.array(0).astype(jnp.int32),
            fuel=jnp.array(self.consts.FUEL_CAPACITY).astype(jnp.float32),
            player=Entity(
                position=jnp.array([12, 78]).astype(jnp.int32),
                direction=jnp.array(4).astype(jnp.int32),
                visibility=jnp.array(1).astype(jnp.int32)
            ),
            dynamite_position=jnp.array([-1, -1]).astype(jnp.int32),  # Inactive at [-1, -1]
            enemy_positions=init_banks_or_police(),
            bank_positions=init_banks_or_police(),
            speed=jnp.array(self.consts.BASE_SPEED).astype(jnp.float32),
            fuel_consumption=jnp.array(1).astype(jnp.float32),
            reserve_speed=jnp.array(0.0).astype(jnp.float32),
            police_reserve_speed=jnp.array(0.0).astype(jnp.float32),
            money=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(self.consts.STARTING_LIVES).astype(jnp.int32),
            fuel_refill=jnp.array(0).astype(jnp.int32),
            obs_stack=None,
            map_collision=self.city_collision_maps[0],
            spawn_points=self.city_spawns[0],
            bank_spawn_timers=jnp.array([1, 1, 1]).astype(jnp.int32),
            police_spawn_timers=jnp.array([-1, -1, -1]).astype(jnp.int32),
            dynamite_timer=jnp.array([-1]).astype(jnp.int32),
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),  # -1 means no pending spawn
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32),  # Bank indices for pending spawns
            game_paused=jnp.array(False).astype(jnp.bool_),
            bank_heists=jnp.array(0).astype(jnp.int32),
            random_key=key,  # Use the provided random key
            time=jnp.array(0, dtype=jnp.int32),
        )
        obs = self._get_observation(state)
        return  obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def validate_input(self, state: BankHeistState, player: Entity, input: jnp.ndarray) -> Entity:
        """
        Confirm that the player is not trying to move into a wall.

        Returns:
            EntityPosition: Contains the new direction of the player after validating the input.
        """
        new_position = self.move(player, input)
        new_position = new_position.replace(direction=input)
        collision = self.check_background_collision(state, new_position)
        direction = jax.lax.cond(collision >= 255,
            lambda: player.direction,
            lambda: new_position.direction
        )
        return player.replace(direction=direction), collision >= 255

    @partial(jax.jit, static_argnums=(0,))
    def check_background_collision(self, state: BankHeistState, new_position: Entity) -> int:
        """
        Check for collisions with the background (walls, portals).

        Returns:
            int: The maximum collision value found(255: wall, 100: portal, 200: exit, 0: empty space).
        """
        new_coords = jnp.array([new_position.position[0], new_position.position[1]-1])
        new_position_bg: jnp.ndarray = jax.lax.dynamic_slice(operand=state.map_collision,
                          start_indices=new_coords, slice_sizes=self.consts.COLLISION_BOX)
        max_value = jnp.max(new_position_bg)
        return max_value

    @partial(jax.jit, static_argnums=(0,))
    def portal_handler(self, entity: Entity, collision: int, is_police: bool) -> Entity:
        """
        Handle portal collisions by moving the entity to the corresponding portal exit.

        Returns:
            Entity: The new position of the entity after handling the portal collision.
        """
        side = entity.position[0] <= 80
        side = side.astype(int)
        
        police_hit_exit = jnp.logical_and(is_police, collision == 200)
        portal_collision = jnp.logical_or(collision == 100, police_hit_exit)

        new_position = jax.lax.cond(portal_collision,
            lambda: entity.replace(position=jnp.array([self.consts.PORTAL_X[side], entity.position[1]])),
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
        new_banks = state.bank_positions.replace(visibility=new_bank_visibility)
        
        # Find an available pending spawn slot
        available_spawn_slots = state.pending_police_spawns < 0
        slot_index = jnp.where(available_spawn_slots, jnp.arange(len(available_spawn_slots)), len(available_spawn_slots))
        first_available_slot = jnp.min(slot_index)
        first_available_slot = jnp.where(first_available_slot >= len(available_spawn_slots), 0, first_available_slot)
        
        # Set up delayed spawn using constant
        new_pending_spawns = state.pending_police_spawns.at[first_available_slot].set(self.consts.POLICE_SPAWN_DELAY)
        new_pending_bank_indices = state.pending_police_bank_indices.at[first_available_slot].set(bank_hit_index)

        # Increase bank heists count by 1
        new_bank_heists = state.bank_heists + 1

        money_bonus = jnp.minimum(new_bank_heists, 9)
        # Increase score by bank robbery reward
        new_money = state.money + (self.consts.BASE_BANK_ROBBERY_REWARD * money_bonus)

        return state.replace(
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
        new_player = state.player.replace(position=default_player_position)

        # Reset all police cars to spawn point (76, 132)
        reset_police_position = jnp.array([76, 132]).astype(jnp.int32)
        # Create array with reset position for all police cars
        reset_positions = jnp.tile(reset_police_position[None, :], (state.enemy_positions.position.shape[0], 1))
        # Only reset visible police cars, keep invisible ones at their current position
        visible_mask = state.enemy_positions.visibility > 0
        new_police_positions = jnp.where(visible_mask[:, None], reset_positions, state.enemy_positions.position)
        new_police = state.enemy_positions.replace(position=new_police_positions)
        
        return state.replace(
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
        # Get spawn position from bank index
        spawn_position = state.bank_positions.position[bank_index]
        
        # Update police positions
        new_positions = state.enemy_positions.position.at[bank_index].set(spawn_position)
        new_directions = state.enemy_positions.direction.at[bank_index].set(4)  # Default direction
        new_visibility = state.enemy_positions.visibility.at[bank_index].set(1)  # Make visible

        new_police = state.enemy_positions.replace(
            position=new_positions,
            direction=new_directions, 
            visibility=new_visibility
        )
        
        return state.replace(enemy_positions=new_police)

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
                return spawned_state.replace(
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
            new_dynamite_timer = jnp.array([self.consts.DYNAMITE_EXPLOSION_DELAY]).astype(jnp.int32)  # 90 frames until explosion
            return state.replace(
                dynamite_position=new_dynamite_position,
                dynamite_timer=new_dynamite_timer,
                fuel=state.fuel - self.consts.DYNAMITE_COST
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
        total_score_added = jnp.array(0)
        
        def check_and_kill_police(i, carry):
            current_state, killed_count, score_added = carry
            police_pos = current_state.enemy_positions.position[i]
            police_visible = current_state.enemy_positions.visibility[i] > 0
            
            # Calculate distance between dynamite and police car
            distance = jnp.linalg.norm(dynamite_pos - police_pos)
            
            # Kill police car if it's visible and within 5x5 range (distance <= ~3.5 pixels)
            # Using 2.5 as threshold for a 5x5 area (half the diagonal of 5x5 square)
            within_range = distance <= 5
            should_kill = police_visible & within_range
            
            # Calculate score before killing this police car
            # Count inactive police cars (visibility == 0) before this kill
            inactive_count = jnp.sum(current_state.enemy_positions.visibility == 0)
            # Use JAX where operations to select reward based on inactive count
            # 0 inactive (3 active) -> 50 points, 1 inactive (2 active) -> 30 points, 2+ inactive (1 active) -> 10 points
            kill_reward = jnp.where(inactive_count == 0, self.consts.POLICE_KILL_REWARD[0],
                         jnp.where(inactive_count == 1, self.consts.POLICE_KILL_REWARD[1], self.consts.POLICE_KILL_REWARD[2]))
            
            # Set visibility to 0 (kill) if within range
            new_visibility = current_state.enemy_positions.visibility.at[i].set(
                jnp.where(should_kill, 0, current_state.enemy_positions.visibility[i])
            )
            # Set respawn timer for corresponding Bank
            new_respawn_timer = current_state.bank_spawn_timers.at[i].set(
                jnp.where(should_kill, self.consts.BANK_RESPAWN_TIME, current_state.bank_spawn_timers[i])
            )

            new_police = current_state.enemy_positions.replace(visibility=new_visibility)
            updated_state = current_state.replace(enemy_positions=new_police, bank_spawn_timers=new_respawn_timer)
            
            # Increment killed count and add score if a police car was killed
            new_killed_count = killed_count + jnp.where(should_kill, 1, 0)
            new_score_added = score_added + jnp.where(should_kill, kill_reward, 0)
            
            return (updated_state, new_killed_count, new_score_added)
        
        # Process all police cars and count kills
        final_state, total_killed, total_score = jax.lax.fori_loop(
            0, len(state.enemy_positions.visibility), 
            check_and_kill_police, 
            (state, police_killed_count, total_score_added)
        )
        
        # Add accumulated score for killed police cars
        new_money = final_state.money + total_score

        return final_state.replace(money=new_money)
    
    @partial(jax.jit, static_argnums=(0,))
    def check_player_in_explosion_range(self, state: BankHeistState) -> chex.Array:
        """
        Check if the player is within the 5x5 explosion range of the dynamite.

        Args:
            state: Current game state

        Returns:
            chex.Array: Boolean indicating if player is within explosion range
        """
        dynamite_pos = state.dynamite_position
        player_pos = state.player.position
        
        # Calculate distance between dynamite and player
        distance = jnp.linalg.norm(dynamite_pos - player_pos)
        
        # Check if player is within 5x5 range (distance <= 5, same as police cars)
        within_range = distance <= 5
        
        return within_range
    
    @partial(jax.jit, static_argnums=(0,))
    def explode_dynamite(self, state: BankHeistState) -> BankHeistState:
        """
        Handle dynamite explosion - kill police cars within 5x5 range, check player collision, and make dynamite inactive.

        Args:
            state: Current game state

        Returns:
            BankHeistState: Updated state with police cars killed, player life decreased if in range, and dynamite deactivated
        """
        # Kill police cars within 5x5 range of dynamite explosion
        updated_state = self.kill_police_in_explosion_range(state)
        
        # Check if player is within explosion range and handle like police collision
        player_in_range = self.check_player_in_explosion_range(updated_state)
        updated_state = jax.lax.cond(player_in_range, lambda: self.lose_life(updated_state), lambda: updated_state)
        
        # Deactivate the dynamite by setting position to [-1, -1] and timer to -1
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        
        return updated_state.replace(
            dynamite_position=new_dynamite_position,
            dynamite_timer=new_dynamite_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: BankHeistState) -> BankHeistState:
        new_level = state.level+1
        new_difficulty_level = jnp.minimum(new_level // self.consts.CITIES_PER_LEVEL, self.consts.MAX_LEVEL-1)
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player.replace(position=default_player_position)
        empty_police = init_banks_or_police()
        empty_banks = init_banks_or_police()
        new_speed = self.consts.BASE_SPEED * jnp.power(self.consts.SPEED_INCREASE_PER_LEVEL, new_difficulty_level)
        new_fuel_consumption = jnp.power(self.consts.FUEL_CONSUMPTION_INCREASE_PER_LEVEL, new_difficulty_level)
        new_fuel = jnp.maximum(state.fuel, jax.lax.dynamic_index_in_dim(self.consts.REFILL_TABLE, state.bank_heists, axis=0, keepdims=False))
        new_fuel_refill=jnp.array(0).astype(jnp.int32)
        map_id = new_level % len(self.city_collision_maps)
        new_map_collision = jax.lax.dynamic_index_in_dim(self.city_collision_maps, map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(self.city_spawns, map_id, axis=0, keepdims=False)
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)  # Inactive dynamite
        new_bank_spawn_timers = jnp.array([1,1,1]).astype(jnp.int32)
        new_police_spawn_timers = jnp.array([-1,-1,-1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        new_player_lives = jax.lax.cond(state.bank_heists >= 9, lambda: state.player_lives + 1, lambda: state.player_lives)
        
        # Calculate city reward if 9 or more banks were robbed
        # Use level % 4 to cycle through CITY_REWARD indices (0, 1, 2, 3)
        city_reward_index = state.level % 4
        # Use JAX operations to select reward based on index
        # city_reward_index cycles through 0, 1, 2, 3 -> rewards are 93, 186, 279, 372
        city_reward = jnp.where(city_reward_index == 0, self.consts.CITY_REWARD[0],
                      jnp.where(city_reward_index == 1, self.consts.CITY_REWARD[1],
                      jnp.where(city_reward_index == 2, self.consts.CITY_REWARD[2], self.consts.CITY_REWARD[3])))
        city_reward_bonus = jax.lax.cond(
            state.bank_heists >= 9,
            lambda: city_reward,
            lambda: 0
        )
        new_money = state.money + city_reward_bonus + (state.difficulty_level * self.consts.BONUS_REWARD)
        
        return state.replace(
            level=new_level,
            difficulty_level=new_difficulty_level,
            player=new_player,
            player_lives=new_player_lives,
            money=new_money,
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
            lambda: new_position.replace(position=jnp.array([new_position.position[0], new_position.position[1] + 1])),  # DOWN
            lambda: new_position.replace(position=jnp.array([new_position.position[0], new_position.position[1] - 1])),  # UP
            lambda: new_position.replace(position=jnp.array([new_position.position[0] + 1, new_position.position[1]])),  # RIGHT
            lambda: new_position.replace(position=jnp.array([new_position.position[0] - 1, new_position.position[1]])),  # LEFT
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
        all_directions = jnp.array([self.consts.DIR_DOWN, self.consts.DIR_UP, self.consts.DIR_RIGHT, self.consts.DIR_LEFT])
        
        # Calculate reverse direction
        reverse_direction = jax.lax.switch(current_direction, [
            lambda: self.consts.DIR_UP,    # If going DOWN, reverse is UP
            lambda: self.consts.DIR_DOWN,  # If going UP, reverse is DOWN  
            lambda: self.consts.DIR_LEFT,  # If going RIGHT, reverse is LEFT
            lambda: self.consts.DIR_RIGHT, # If going LEFT, reverse is RIGHT
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
            new_positions = jax.vmap(get_new_position)(jnp.array([self.consts.DIR_DOWN, self.consts.DIR_UP, self.consts.DIR_RIGHT, self.consts.DIR_LEFT]))
            
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
                self.consts.POLICE_RANDOM_FACTOR * random_noise + 
                self.consts.POLICE_BIAS_FACTOR * (distance_bias + 0.1)  # Add small constant to prevent zero weights
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
                moved_entity = self.portal_handler(moved_entity, collision, True)
                
                # Update police positions
                new_positions = state_inner.enemy_positions.position.at[i].set(moved_entity.position)
                new_directions = state_inner.enemy_positions.direction.at[i].set(new_direction)
                
                new_police = state_inner.enemy_positions.replace(
                    position=new_positions,
                    direction=new_directions
                )
                
                return state_inner.replace(enemy_positions=new_police)
            
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
        step_random_key, next_random_key  = jax.random.split(state.random_key)
        updated_state = state.replace(random_key=next_random_key)
        return step_random_key, updated_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Get random key for this step and advance state's random key
        step_random_key, new_state = self.advance_random_key(state)
        full_speed = new_state.speed + new_state.reserve_speed
        new_state = new_state.replace(reserve_speed=jnp.mod(full_speed, 1.0))
        full_speed = full_speed.astype(jnp.int32)
        # Player step. Must be run first as this step unpauses the game if it is paused!
        new_state = jax.lax.fori_loop(0, full_speed, lambda i, s: self.player_step(s, atari_action), new_state)
        police_speed = jax.lax.cond(
            new_state.game_paused,
            lambda: jnp.array(0).astype(jnp.float32),
            lambda: (new_state.speed * self.consts.POLICE_SLOW_DOWN_FACTOR + state.police_reserve_speed)        )
        new_state = new_state.replace(police_reserve_speed=jnp.mod(police_speed, 1.0))
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

        
        # Game is done when player runs out of lives
        done = new_state.player_lives < 0
        
        # Get new observation and update obs_stack
        obs = self._get_observation(new_state)
        
        # Update observation stack by shifting and adding new observation
        if new_state.obs_stack is not None:
            # Shift the stack and add new observation
            old_stack = new_state.obs_stack
            def shift_and_add(x_old, x_new):
                # Shift existing frames and add new one at the end
                shifted = x_old[1:]  # Remove oldest frame
                x_new_expanded = jnp.expand_dims(x_new, axis=0)
                return jnp.concatenate([shifted, x_new_expanded], axis=0)
            
            new_obs_stack = jax.tree.map(shift_and_add, old_stack, obs)
            new_state = new_state.replace(obs_stack=new_obs_stack)
        reward = new_state.money - state.money
        new_time = (state.time + 1).astype(jnp.int32)
        new_state = new_state.replace(time=new_time)

        # Create info
        info = BankHeistInfo(
            time=new_state.time,
        )

        
        return self._get_observation(new_state), new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: BankHeistState, atari_action: chex.Array) -> BankHeistState:
        """
        Handles player Input & movement.

        Returns:
            BankHeistState: The new state of the game after the player's action.
        """
        # Convert Atari action to internal direction code
        # Map Atari actions (Action.UP=2, Action.DOWN=5, etc.) to internal directions (DIR_UP=1, DIR_DOWN=0, etc.)
        # Also handle fire actions and diagonal actions
        internal_direction = jnp.select(
            condlist=[
                # Fire actions - extract base direction
                atari_action == Action.UPFIRE,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                # Cardinal directions
                atari_action == Action.UP,
                atari_action == Action.DOWN,
                atari_action == Action.LEFT,
                atari_action == Action.RIGHT,
                # Diagonal directions - map to primary vertical component
                atari_action == Action.UPRIGHT,
                atari_action == Action.UPLEFT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.DOWNLEFT,
            ],
            choicelist=[
                self.consts.DIR_UP,      # UPFIRE -> UP
                self.consts.DIR_DOWN,    # DOWNFIRE -> DOWN
                self.consts.DIR_LEFT,    # LEFTFIRE -> LEFT
                self.consts.DIR_RIGHT,   # RIGHTFIRE -> RIGHT
                self.consts.DIR_UP,      # UPRIGHTFIRE -> UP (primary vertical)
                self.consts.DIR_UP,      # UPLEFTFIRE -> UP (primary vertical)
                self.consts.DIR_DOWN,    # DOWNRIGHTFIRE -> DOWN (primary vertical)
                self.consts.DIR_DOWN,    # DOWNLEFTFIRE -> DOWN (primary vertical)
                self.consts.DIR_UP,      # UP -> UP
                self.consts.DIR_DOWN,    # DOWN -> DOWN
                self.consts.DIR_LEFT,    # LEFT -> LEFT
                self.consts.DIR_RIGHT,   # RIGHT -> RIGHT
                self.consts.DIR_UP,      # UPRIGHT -> UP (primary vertical)
                self.consts.DIR_UP,      # UPLEFT -> UP (primary vertical)
                self.consts.DIR_DOWN,    # DOWNRIGHT -> DOWN (primary vertical)
                self.consts.DIR_DOWN,    # DOWNLEFT -> DOWN (primary vertical)
            ],
            default=self.consts.DIR_NOOP,  # NOOP, FIRE, or unknown -> NOOP
        )
        
        # For NOOP and FIRE, keep current direction
        player_input = jnp.where(
            jnp.logical_or(atari_action == Action.NOOP, atari_action == Action.FIRE),
            state.player.direction,
            internal_direction
        )
        
        current_player, invalid_input = self.validate_input(state, state.player, player_input)
        new_player = self.move(current_player, current_player.direction)
        collision = self.check_background_collision(state, new_player)
        new_player = jax.lax.cond(collision >= 255,
            lambda: current_player,
            lambda: new_player
        )
        new_player = self.portal_handler(new_player, collision, False)

        new_state = state.replace(
            player=new_player,
            game_paused=jnp.where(jnp.logical_and(jnp.logical_not(invalid_input), atari_action != Action.NOOP), jnp.array(False).astype(jnp.bool_), state.game_paused)
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
        # Check if action is a fire action (any action with FIRE in it)
        is_fire = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(atari_action == Action.FIRE, atari_action == Action.UPFIRE),
                jnp.logical_or(atari_action == Action.DOWNFIRE, atari_action == Action.LEFTFIRE)
            ),
            jnp.logical_or(
                jnp.logical_or(atari_action == Action.RIGHTFIRE, atari_action == Action.UPRIGHTFIRE),
                jnp.logical_or(atari_action == Action.UPLEFTFIRE, jnp.logical_or(atari_action == Action.DOWNRIGHTFIRE, atari_action == Action.DOWNLEFTFIRE))
            )
        )
        new_state = jax.lax.cond(is_fire, 
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

        new_state = state.replace(fuel=new_fuel)
        new_state = jax.lax.cond(
            new_state.fuel <= 0,
            lambda: self.lose_life(new_state).replace(fuel=self.consts.REVIVAL_FUEL),
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

            new_banks = state.bank_positions.replace(position=new_bank_positions, visibility=new_visibility)
            return state.replace(bank_positions=new_banks)

        new_bank_spawn_timers = jnp.where(state.bank_spawn_timers >= 0, state.bank_spawn_timers - 1, state.bank_spawn_timers)
        new_police_spawn_timers = jnp.where(state.police_spawn_timers >= 0, state.police_spawn_timers - 1, state.police_spawn_timers)
        new_dynamite_timer = jnp.where(state.dynamite_timer >= 0, state.dynamite_timer - 1, state.dynamite_timer)
        
        # Handle pending police spawns
        new_pending_police_spawns = jnp.where(state.pending_police_spawns >= 0, state.pending_police_spawns - 1, state.pending_police_spawns)

        new_state = state.replace(
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

        dir_to_angle = jnp.array([180.0, 0.0, 90.0, 270.0, 0.0], dtype=jnp.float32)

        player = ObjectObservation.create(
            x=jnp.clip(state.player.position[0], 0, self.consts.WIDTH - 1),
            y=jnp.clip(state.player.position[1], 0, self.consts.HEIGHT - 1),
            width=jnp.array(self.consts.COLLISION_BOX[0], dtype=jnp.int32),
            height=jnp.array(self.consts.COLLISION_BOX[1], dtype=jnp.int32),
            orientation=dir_to_angle[state.player.direction],
            active=state.player.visibility
        )
        
        enemies = ObjectObservation.create(
            x=jnp.clip(state.enemy_positions.position[:, 0], 0, self.consts.WIDTH - 1),
            y=jnp.clip(state.enemy_positions.position[:, 1], 0, self.consts.HEIGHT - 1),
            width=jnp.full((3,), self.consts.COLLISION_BOX[0], dtype=jnp.int32),
            height=jnp.full((3,), self.consts.COLLISION_BOX[1], dtype=jnp.int32),
            orientation=dir_to_angle[state.enemy_positions.direction],
            active=state.enemy_positions.visibility
        )

        banks = ObjectObservation.create(
            x=jnp.clip(state.bank_positions.position[:, 0], 0, self.consts.WIDTH - 1),
            y=jnp.clip(state.bank_positions.position[:, 1], 0, self.consts.HEIGHT - 1),
            width=jnp.full((3,), self.consts.COLLISION_BOX[0], dtype=jnp.int32),
            height=jnp.full((3,), self.consts.COLLISION_BOX[1], dtype=jnp.int32),
            orientation=dir_to_angle[state.bank_positions.direction],
            active=state.bank_positions.visibility
        )
        
        # Determine if dynamite is active (not at [-1, -1])
        dynamite_active = jnp.logical_not(jnp.all(state.dynamite_position == jnp.array([-1, -1])))
        dynamite = ObjectObservation.create(
            x=jnp.clip(state.dynamite_position[0], 0, self.consts.WIDTH - 1),
            y=jnp.clip(state.dynamite_position[1], 0, self.consts.HEIGHT - 1),
            width=jnp.array(self.consts.COLLISION_BOX[0], dtype=jnp.int32),
            height=jnp.array(self.consts.COLLISION_BOX[1], dtype=jnp.int32),
            active=dynamite_active.astype(jnp.int32)
        )

        return BankHeistObservation(
            player=player,
            enemies=enemies,
            banks=banks,
            dynamite=dynamite,
            fuel=state.fuel,
            fuel_refill=state.fuel_refill,
            lives=state.player_lives,
            score=state.money
        )

    def render(self, state: BankHeistState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BankHeistInfo) -> BankHeistInfo:
        return BankHeistInfo(time=state.time)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BankHeistState, current_state: BankHeistState) -> chex.Array:
        return current_state.money - previous_state.money
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BankHeistState) -> bool:
        return state.player_lives < 0

class BankHeistRenderer(JAXGameRenderer):
    def __init__(self, consts: BankHeistConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or BankHeistConstants()
        super().__init__(self.consts)
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 2. Define asset path
        sprite_path = self.consts.SPRITES_DIR
        # 3. Load all assets using the manifest from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        city_asset = next((a for a in final_asset_config if a['name'] == 'cities'), None)
        if city_asset:
            final_asset_config.remove(city_asset)
            city_files = city_asset['files']
            # Add first city as the default background
            final_asset_config.append({'name': 'background', 'type': 'background', 'file': city_files[0]})
            # Add all cities (including first) as a group for indexing
            final_asset_config.append({'name': 'city_maps', 'type': 'group', 'files': city_files})
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # --- 4. Manual Padding for Cars (to match old symmetric padding) ---
        p_side_mask = self.SHAPE_MASKS['player_side']   # (H, W_side)
        p_front_mask = self.SHAPE_MASKS['player_front'] # (H, W_front)
        
        pad_w = p_side_mask.shape[1] - p_front_mask.shape[1]
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        p_front_padded = jnp.pad(
            p_front_mask, 
            ((0,0), (pad_left, pad_right)), 
            mode="constant", 
            constant_values=self.jr.TRANSPARENT_ID
        )
        self.PLAYER_MASKS = jnp.stack([p_side_mask, p_front_padded])
        
        pol_side_mask = self.SHAPE_MASKS['police_side']
        pol_front_mask = self.SHAPE_MASKS['police_front']
        pad_w_pol = pol_side_mask.shape[1] - pol_front_mask.shape[1]
        pad_left_pol = pad_w_pol // 2
        pad_right_pol = pad_w_pol - pad_left_pol
        pol_front_padded = jnp.pad(
            pol_front_mask,
            ((0,0), (pad_left_pol, pad_right_pol)),
            mode="constant",
            constant_values=self.jr.TRANSPARENT_ID
        )
        self.POLICE_MASKS = jnp.stack([pol_side_mask, pol_front_padded])
        
        # --- 5. Store procedural color ID ---
        self.fuel_color_id = self.COLOR_TO_ID[(167, 26, 26)]
    @partial(jax.jit, static_argnums=(0,))
    def _render_fuel_tank(self, raster, state):
        """
        Renders the fuel tank base and procedurally fills it based on fuel level.
        This replaces the old 'fuel_tank_filler' logic.
        """
        # 1. Draw the base tank sprite
        tank_mask = self.SHAPE_MASKS['fuel_tank'] # Shape (11, 25)
        tank_x, tank_y = self.consts.FUEL_TANK_POSITION
        raster = self.jr.render_at(raster, tank_x, tank_y, tank_mask, flip_offset=self.FLIP_OFFSETS['fuel_tank'])
        # 2. Calculate fill
        level = state.fuel / self.consts.FUEL_CAPACITY
        
        # Get scaled coords
        scaled_tank_x = jnp.round(tank_x * self.jr.config.width_scaling).astype(jnp.int32)
        scaled_tank_y = jnp.round(tank_y * self.jr.config.height_scaling).astype(jnp.int32)
        
        # Get scaled sprite dimensions from the mask
        sprite_h, sprite_w = tank_mask.shape # (11, 25)
        
        # We calculate the fill level based on the sprite's HEIGHT (sprite_h)
        # instead of its width (sprite_w).
        pixel_level = jnp.ceil(level * sprite_h).astype(jnp.int32)
        
        # 3. Create fill mask (filling from the bottom-up)
        xx, yy = self.jr._xx, self.jr._yy
        
        # Calculate the top Y coordinate of the filled portion
        fill_top_y = scaled_tank_y + sprite_h - pixel_level
        
        # Bounding box for the fill (fills vertically)
        fill_box_mask = (yy >= fill_top_y) & (yy < scaled_tank_y + sprite_h) & \
                        (xx >= scaled_tank_x) & (xx < scaled_tank_x + sprite_w)
        
        # 4. Get mask for the tank sprite itself (to fill *inside* it)
        # Sample the tank mask at its rendered position
        sprite_coords_y = yy - scaled_tank_y
        sprite_coords_x = xx - scaled_tank_x
        sampled_tank_mask = map_coordinates(
            tank_mask,
            [sprite_coords_y, sprite_coords_x],
            order=0,
            cval=self.jr.TRANSPARENT_ID
        )
        tank_pixels_mask = (sampled_tank_mask != self.jr.TRANSPARENT_ID)
        # 5. Combine masks
        final_fill_mask = fill_box_mask & tank_pixels_mask
        
        # 6. Paint the raster
        return jnp.where(final_fill_mask, jnp.asarray(self.fuel_color_id, raster.dtype), raster)
    @partial(jax.jit, static_argnums=(0,))
    def render_lives(self, raster, state, life_mask):
        # Use the manually padded player side mask
        def body_fun(i, rast):
            return jax.lax.cond(
                i < state.player_lives,
                # Use [0,0] flip_offset for symmetrically padded sprites
                lambda r: self.jr.render_at(r, self.consts.LIFE_POSITIONS[i][0], self.consts.LIFE_POSITIONS[i][1], life_mask, flip_offset=jnp.array([0,0])),
                lambda r: r,
                rast
            )
        raster = jax.lax.fori_loop(0, len(self.consts.LIFE_POSITIONS), body_fun, raster)
        return raster
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Start with a base raster (e.g., black or first city map)
        # We use the 'background' (map_1) as the default
        raster = self.jr.create_object_raster(self.BACKGROUND)
        ### Render City
        # Get the correct city map mask from the 'city_maps' group
        city_mask = self.SHAPE_MASKS['city_maps'][state.level % self.SHAPE_MASKS['city_maps'].shape[0]]
        # Stamp the current city map over the background
        raster = self.jr.render_at(raster, 0, 0, city_mask, flip_offset=self.FLIP_OFFSETS['city_maps'])
        ### Render Player
        # Use the manually padded masks
        branches = [
            lambda: self.PLAYER_MASKS[1],  # DOWN (Front)
            lambda: self.PLAYER_MASKS[1],  # UP (Front)
            lambda: self.PLAYER_MASKS[0],  # RIGHT (Side)
            lambda: jnp.flip(self.PLAYER_MASKS[0], axis=1),   # LEFT (Flipped Side)
        ]
        player_direction = jax.lax.cond(
            state.player.direction == 4, # NOOP
            lambda: 2, # Default to RIGHT
            lambda: state.player.direction
        )
        player_mask = jax.lax.switch(player_direction, branches)
        # Use [0,0] offset for symmetric padding
        raster = self.jr.render_at(raster, state.player.position[0], state.player.position[1], player_mask, flip_offset=jnp.array([0,0]))
        ### Render Fuel Tank (using new procedural function)
        raster = self._render_fuel_tank(raster, state)
        # Render the Fuel gauge
        fuel_gauge_position = jax.lax.dynamic_index_in_dim(self.consts.TANK_LEVELS, state.bank_heists, axis=0, keepdims=False)
        fuel_gauge_mask = self.SHAPE_MASKS['fuel_gauge']
        raster = jax.lax.cond(fuel_gauge_position > 0,
                              lambda: self.jr.render_at(raster, 
                                                        self.consts.FUEL_TANK_POSITION[0]+8, 
                                                        self.consts.FUEL_TANK_POSITION[1]+(self.consts.TANK_HEIGHT - fuel_gauge_position), 
                                                        fuel_gauge_mask,
                                                        flip_offset=self.FLIP_OFFSETS['fuel_gauge']),
                              lambda: raster,
                              )
        ### Render Player Lives
        life_mask = self.PLAYER_MASKS[0] # Player side view
        raster = self.render_lives(raster, state, life_mask)
        ### Render Banks
        bank_mask = self.SHAPE_MASKS['bank']
        bank_flip_offset = self.FLIP_OFFSETS['bank']
        def render_bank(i, r):
            return jax.lax.cond(
                state.bank_positions.visibility[i] != 0,
                lambda r_in: self.jr.render_at(r_in, state.bank_positions.position[i, 0], state.bank_positions.position[i, 1], bank_mask, flip_offset=bank_flip_offset),
                lambda r_in: r_in,
                r
            )
        raster = jax.lax.fori_loop(0, state.bank_positions.position.shape[0], render_bank, raster)
        ### Render Police Cars
        police_branches = [
            lambda: self.POLICE_MASKS[1],  # DOWN (Front)
            lambda: self.POLICE_MASKS[1],  # UP (Front)
            lambda: jnp.flip(self.POLICE_MASKS[0], axis=1),   # RIGHT (Flipped Side)
            lambda: self.POLICE_MASKS[0],   # LEFT (Side)
        ]
        
        def render_police_car(i, r):
            def render_police(raster_input):
                police_direction = jax.lax.cond(
                    state.enemy_positions.direction[i] == 4, # NOOP
                    lambda: 2,  # Default to RIGHT
                    lambda: state.enemy_positions.direction[i]
                )
                police_mask = jax.lax.switch(police_direction, police_branches)
                # Use [0,0] offset for symmetric padding
                return self.jr.render_at(raster_input, state.enemy_positions.position[i, 0], state.enemy_positions.position[i, 1], police_mask, flip_offset=jnp.array([0,0]))
            
            return jax.lax.cond(
                state.enemy_positions.visibility[i] != 0,
                render_police,
                lambda r_in: r_in,
                r
            )
        raster = jax.lax.fori_loop(0, state.enemy_positions.position.shape[0], render_police_car, raster)
        ### Render Dynamite (only when active)
        def render_dynamite(raster_input):
            dynamite_mask = self.SHAPE_MASKS['dynamite']
            return self.jr.render_at(raster_input, state.dynamite_position[0], state.dynamite_position[1], dynamite_mask, flip_offset=self.FLIP_OFFSETS['dynamite'])
        
        dynamite_active = ~jnp.all(state.dynamite_position == jnp.array([-1, -1]))
        raster = jax.lax.cond(dynamite_active, render_dynamite, lambda r: r, raster)
        ### Render Score
        score_digits_arr = self.jr.int_to_digits(state.money, max_digits=4)
        raster = self.jr.render_label_selective(
            raster, 90, 179, 
            all_digits=score_digits_arr, 
            digit_id_masks=self.SHAPE_MASKS['digits'], 
            start_index=0, 
            num_to_render=4, # Always render 4 digits for score
            spacing=12,
            max_digits_to_render=4
        )
        # Final conversion from ID raster to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)