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
        {'name': 'fuel_tank', 'type': 'single', 'file': 'fuel_tank.npy'}, 
        # This file is (3, 1, 4) on disk but should be rendered as (1, 3).
        {'name': 'fuel_gauge', 'type': 'single', 'file': 'fuel_gauge.npy', 'transpose': True},
        
        # UI
        {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        
        # Bank Scores
        {'name': 'bank_scores', 'type': 'group', 'files': [f"{i}0.npy" for i in range(1, 10)]},
        
        # Procedural
        {'name': 'fuel_color', 'type': 'procedural', 'data': fuel_color_rgba},
    )
    
    return config

class BankHeistConstants(AutoDerivedConstants):
    # Basic dimensions
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    
    # Internal direction codes
    DIR_DOWN: int = struct.field(pytree_node=False, default=0)
    DIR_UP: int = struct.field(pytree_node=False, default=1)
    DIR_RIGHT: int = struct.field(pytree_node=False, default=2)
    DIR_LEFT: int = struct.field(pytree_node=False, default=3)
    DIR_NOOP: int = struct.field(pytree_node=False, default=4)
    
    # Fuel constants
    FUEL_CAPACITY: float = struct.field(pytree_node=False, default=10200.0)
    TANK_HEIGHT: float = struct.field(pytree_node=False, default=25.0)
    TANK_LEVELS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 2, 4, 6, 9, 12, 16, 18, 25]))
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
    
    LIFE_POSITIONS: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    
    # Reward constants
    BASE_BANK_ROBBERY_REWARD: int = struct.field(pytree_node=False, default=10)
    POLICE_KILL_REWARD: Tuple[int, int, int] = struct.field(pytree_node=False, default=(50, 30, 10))
    CITY_REWARD: Tuple[int, int, int, int] = struct.field(pytree_node=False, default=(93, 186, 279, 372))
    BONUS_REWARD: int = struct.field(pytree_node=False, default=372)

    # The deterministic 16-step flat array from ALE
    HARDCODED_BANK_SPAWNS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                [68, 94],  # 0
                [16, 102],  # 1
                [136, 78],  # 2
                [56, 142],  # 3
                [136, 134],  # 4
                [76, 134],  # 5
                [116, 62],  # 6
                [16, 166],  # 7
                [136, 54],  # 8
                [128, 86],  # 9
                [84, 166],  # 10
                [84, 46],  # 11
                [32, 46],  # 12
                [36, 126],  # 13
                [84, 110],  # 14
                [48, 70],  # 15
            ],
            dtype=jnp.int32,
        ),
    )
    
    # Position constants
    FUEL_TANK_POSITION: Tuple[int, int] = struct.field(pytree_node=False, default=(42, 12))
    COLLISION_BOX: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 8))
    PORTAL_X: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([17, 140]))
    LEVEL_TRANSITION_SPAWN: Tuple[int, int] = struct.field(pytree_node=False, default=(13, 78))
    
    WINDOW_WIDTH: Optional[int] = struct.field(pytree_node=False, default=None)
    WINDOW_HEIGHT: Optional[int] = struct.field(pytree_node=False, default=None)
    
    # Police constants
    POLICE_SLOW_DOWN_FACTOR: float = struct.field(pytree_node=False, default=0.9)
    BANK_RESPAWN_TIME: int = struct.field(pytree_node=False, default=300)
    POLICE_SPAWN_DELAY: int = struct.field(pytree_node=False, default=120)
    POLICE_RANDOM_FACTOR: float = struct.field(pytree_node=False, default=0.7)
    POLICE_BIAS_FACTOR: float = struct.field(pytree_node=False, default=0.4)
    # Difficulty-based police spawn delays (frames) after each bank robbery
    SPAWN_DELAYS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([150, 118, 102, 78, 46, 30, 14]),
    )
    
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

class CityPersistentState(struct.PyTreeNode):
    bank_positions: Entity
    bank_spawn_timers: jnp.ndarray
    bank_spawn_indices: jnp.ndarray
    bank_heists: jnp.ndarray

class BankHeistState(struct.PyTreeNode):
    level: chex.Array
    map_id: chex.Array
    difficulty_level: chex.Array
    player: Entity
    player_move_direction: chex.Array
    latched_action: chex.Array
    portal_pending: chex.Array
    portal_pending_side: chex.Array
    dynamite_position: chex.Array
    enemy_positions: Entity
    bank_positions: Entity
    speed: chex.Array
    fuel_consumption: chex.Array
    reserve_speed: chex.Array 
    police_reserve_speed: chex.Array 
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
    pending_police_bank_indices: chex.Array  # Police slot index for each pending spawn
    pending_police_spawn_positions: chex.Array  # (3, 2) position captured at robbery time
    pending_police_scores: chex.Array  # Bank robbery score for the pending spawn
    killed_police_scores: chex.Array  # Score when police is killed, displayed while bank_spawn_timers is high
    game_paused: chex.Array  # Game paused state Is set at beginning and after life was lost
    pending_exit: chex.Array  # Whether a level exit has been requested (one-frame delay)
    bank_heists: chex.Array  # number of bank heists completed in the current level
    explosion_timer: chex.Array  # Timer for the screen flash effect when dynamite explodes
    last_raw_action: chex.Array
    current_diagonal_priority: chex.Array
    total_banks_robbed: chex.Array  # global counter of banks robbed (for difficulty)
    bank_spawn_indices: chex.Array  # 3 independent indices into HARDCODED_BANK_SPAWNS
    random_key: chex.PRNGKey  # Persistent random key that advances each step
    time: chex.Array
    city_states: CityPersistentState

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
        self.city_collision_maps = jnp.array([
            load_city_collision_map(f"map_{i+1}_collision.npy", self.consts.SPRITES_DIR) 
            for i in range(8)
        ])
        self.city_spawns = get_spawn_points(self.city_collision_maps)
        self.renderer = BankHeistRenderer(self.consts)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))
    
    def observation_space(self) -> spaces.Dict:
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
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)
    
    def _init_city_states(self) -> CityPersistentState:
        num_maps = 8
        positions = jnp.zeros((num_maps, 3, 2), dtype=jnp.int32)
        directions = jnp.full((num_maps, 3), 4, dtype=jnp.int32)
        visibilities = jnp.zeros((num_maps, 3), dtype=jnp.int32)
        
        bank_positions = Entity(position=positions, direction=directions, visibility=visibilities)
        bank_spawn_timers = jnp.ones((num_maps, 3), dtype=jnp.int32)
        bank_spawn_indices = jnp.tile(jnp.array([0, 5, 10], dtype=jnp.int32), (num_maps, 1))
        bank_heists = jnp.zeros((num_maps,), dtype=jnp.int32)
        
        return CityPersistentState(
            bank_positions=bank_positions,
            bank_spawn_timers=bank_spawn_timers,
            bank_spawn_indices=bank_spawn_indices,
            bank_heists=bank_heists
        )

    @partial(jax.jit, static_argnums=(0,))
    def save_city_state(self, city_states: CityPersistentState, map_id: int, bank_positions: Entity, bank_spawn_timers: jnp.ndarray, bank_spawn_indices: jnp.ndarray, bank_heists: jnp.ndarray) -> CityPersistentState:
        return city_states.replace(
            bank_positions=city_states.bank_positions.replace(
                position=city_states.bank_positions.position.at[map_id].set(bank_positions.position),
                direction=city_states.bank_positions.direction.at[map_id].set(bank_positions.direction),
                visibility=city_states.bank_positions.visibility.at[map_id].set(bank_positions.visibility),
            ),
            bank_spawn_timers=city_states.bank_spawn_timers.at[map_id].set(bank_spawn_timers),
            bank_spawn_indices=city_states.bank_spawn_indices.at[map_id].set(bank_spawn_indices),
            bank_heists=city_states.bank_heists.at[map_id].set(bank_heists),
        )

    @partial(jax.jit, static_argnums=(0,))
    def load_city_state(self, state: BankHeistState, map_id: int) -> BankHeistState:
        city_states = state.city_states
        return state.replace(
            bank_positions=Entity(
                position=city_states.bank_positions.position[map_id],
                direction=city_states.bank_positions.direction[map_id],
                visibility=city_states.bank_positions.visibility[map_id],
            ),
            bank_spawn_timers=city_states.bank_spawn_timers[map_id],
            bank_spawn_indices=city_states.bank_spawn_indices[map_id],
            bank_heists=city_states.bank_heists[map_id],
        )

    def reset(self, key: chex.PRNGKey) -> BankHeistState:
        initial_player = Entity(
            position=jnp.array([12, 78]).astype(jnp.int32),
            direction=jnp.array(4).astype(jnp.int32),
            visibility=jnp.array(1).astype(jnp.int32)
        )
        state = BankHeistState(
            level=jnp.array(0).astype(jnp.int32),
            map_id=jnp.array(0).astype(jnp.int32),
            difficulty_level=jnp.array(0).astype(jnp.int32),
            fuel=jnp.array(self.consts.FUEL_CAPACITY).astype(jnp.float32),
            player=initial_player,
            player_move_direction=initial_player.direction,
            latched_action=jnp.array(Action.NOOP).astype(jnp.int32),
            portal_pending=jnp.array(False).astype(jnp.bool_),
            portal_pending_side=jnp.array(-1).astype(jnp.int32),
            dynamite_position=jnp.array([-1, -1]).astype(jnp.int32),
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
            pending_police_scores=jnp.array([-1, -1, -1]).astype(jnp.int32),  # Score for the pending spawn
            killed_police_scores=jnp.array([-1, -1, -1]).astype(jnp.int32),  # Score when police is killed
            pending_police_spawn_positions=jnp.full((3, 2), -1).astype(jnp.int32),
            game_paused=jnp.array(False).astype(jnp.bool_),
            pending_exit=jnp.array(False).astype(jnp.bool_),
            bank_heists=jnp.array(0).astype(jnp.int32),
            explosion_timer=jnp.array(0).astype(jnp.int32),
            last_raw_action=jnp.array(Action.NOOP).astype(jnp.int32),
            current_diagonal_priority=jnp.array(self.consts.DIR_UP).astype(jnp.int32),
            total_banks_robbed=jnp.array(0).astype(jnp.int32),
            bank_spawn_indices=jnp.array([0, 5, 10]).astype(jnp.int32),
            random_key=key,  # Use the provided random key
            time=jnp.array(0, dtype=jnp.int32),
            city_states=self._init_city_states()
        )
        obs = self._get_observation(state)
        return  obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def validate_input(self, state: BankHeistState, player: Entity, input: jnp.ndarray) -> Entity:
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
        new_coords = jnp.array([new_position.position[0], new_position.position[1]-1])
        new_position_bg: jnp.ndarray = jax.lax.dynamic_slice(operand=state.map_collision,
                          start_indices=new_coords, slice_sizes=self.consts.COLLISION_BOX)
        max_value = jnp.max(new_position_bg)
        return max_value

    @partial(jax.jit, static_argnums=(0,))
    def portal_handler(self, entity: Entity, collision: int, is_police: bool) -> Entity:
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
        player_pos = player.position
        bank_positions = banks.position
        
        distances = jnp.linalg.norm(bank_positions - player_pos[None, :], axis=1)
        collision_distance = 8 
        
        visible_mask = banks.visibility > 0
        collision_mask = (distances < collision_distance) & visible_mask
        
        bank_index = jnp.where(collision_mask, jnp.arange(len(collision_mask)), -1)
        first_bank_hit = jnp.max(bank_index)
        return collision_mask, first_bank_hit

    @partial(jax.jit, static_argnums=(0,))
    def check_police_collision(self, player: Entity, police: Entity) -> Tuple[chex.Array, chex.Array]:
        player_pos = player.position
        police_positions = police.position
        
        distances = jnp.linalg.norm(police_positions - player_pos[None, :], axis=1)
        collision_distance = 8 
        
        visible_mask = police.visibility > 0
        collision_mask = (distances < collision_distance) & visible_mask
        
        police_index = jnp.where(collision_mask, jnp.arange(len(collision_mask)), -1)
        first_police_hit = jnp.max(police_index)
        return collision_mask, first_police_hit

    @partial(jax.jit, static_argnums=(0,))
    def handle_bank_robbery(self, state: BankHeistState, bank_hit_index: chex.Array) -> BankHeistState:
        # Hide the robbed bank
        new_bank_visibility = state.bank_positions.visibility.at[bank_hit_index].set(0)
        new_banks = state.bank_positions.replace(visibility=new_bank_visibility)
        
        # Find an available pending spawn slot
        available_spawn_slots = state.pending_police_spawns < 0
        slot_index = jnp.where(available_spawn_slots, jnp.arange(len(available_spawn_slots)), len(available_spawn_slots))
        first_available_slot = jnp.min(slot_index)
        first_available_slot = jnp.where(first_available_slot >= len(available_spawn_slots), 0, first_available_slot)
        
        # EXACT ALE SPAWN MATH: 158 base - (8 * total_banks_robbed), floored at 14
        calculated_delay = 158 - (8 * state.total_banks_robbed)
        calculated_delay = jnp.maximum(14, calculated_delay)
        
        # Capture the exact robbery position NOW so the cop always spawns here
        robbery_position = state.bank_positions.position[bank_hit_index]
        new_pending_spawns = state.pending_police_spawns.at[first_available_slot].set(calculated_delay)
        new_pending_bank_indices = state.pending_police_bank_indices.at[first_available_slot].set(bank_hit_index)
        new_pending_spawn_positions = state.pending_police_spawn_positions.at[first_available_slot].set(robbery_position)

        new_bank_heists = state.bank_heists + 1
        new_total_banks_robbed = state.total_banks_robbed + 1

        money_bonus = jnp.minimum(new_bank_heists, 9)
        new_pending_scores = state.pending_police_scores.at[first_available_slot].set(money_bonus)

        # Increase score by bank robbery reward
        new_money = state.money + (self.consts.BASE_BANK_ROBBERY_REWARD * money_bonus)

        return state.replace(
            bank_positions=new_banks,
            pending_police_spawns=new_pending_spawns,
            pending_police_bank_indices=new_pending_bank_indices,
            pending_police_scores=new_pending_scores,
            pending_police_spawn_positions=new_pending_spawn_positions,
            bank_heists=new_bank_heists,
            total_banks_robbed=new_total_banks_robbed,
            money=new_money,
        )

    @partial(jax.jit, static_argnums=(0,))
    def lose_life(self, state: BankHeistState) -> BankHeistState:
        # Reduce player lives by 1
        new_player_lives = state.player_lives - 1
        
        # Reset player to default position
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player.replace(position=default_player_position)

        # Reset all police cars to spawn point
        reset_police_position = jnp.array([76, 132]).astype(jnp.int32)
        reset_positions = jnp.tile(reset_police_position[None, :], (state.enemy_positions.position.shape[0], 1))
        visible_mask = state.enemy_positions.visibility > 0
        new_police_positions = jnp.where(visible_mask[:, None], reset_positions, state.enemy_positions.position)
        new_police = state.enemy_positions.replace(position=new_police_positions)
        
        # Cap difficulty upon death (Grants the 38-frame pity timer)
        new_total_banks_robbed = jnp.minimum(state.total_banks_robbed, jnp.array(15).astype(jnp.int32))

        return state.replace(
            player_lives=new_player_lives,
            player=new_player,
            player_move_direction=new_player.direction,
            portal_pending=jnp.array(False).astype(jnp.bool_),
            portal_pending_side=jnp.array(-1).astype(jnp.int32),
            enemy_positions=new_police,
            total_banks_robbed=new_total_banks_robbed,
            game_paused=jnp.array(True),
            pending_exit=jnp.array(False).astype(jnp.bool_),
        )

    @partial(jax.jit, static_argnums=(0,))
    def spawn_police_car(self, state: BankHeistState, police_slot: chex.Array, spawn_position: chex.Array) -> BankHeistState:
        new_positions = state.enemy_positions.position.at[police_slot].set(spawn_position)
        new_directions = state.enemy_positions.direction.at[police_slot].set(4)
        new_visibility = state.enemy_positions.visibility.at[police_slot].set(1)

        new_police = state.enemy_positions.replace(
            position=new_positions,
            direction=new_directions, 
            visibility=new_visibility
        )
        return state.replace(enemy_positions=new_police)

    @partial(jax.jit, static_argnums=(0,))
    def process_pending_police_spawns(self, state: BankHeistState) -> BankHeistState:
        def process_single_spawn(i, current_state):
            def spawn_at_bank_index(state_inner):
                police_slot = state_inner.pending_police_bank_indices[i]
                spawn_pos = state_inner.pending_police_spawn_positions[i]
                spawned_state = self.spawn_police_car(state_inner, police_slot, spawn_pos)
                new_pending_spawns = spawned_state.pending_police_spawns.at[i].set(-1)
                new_pending_indices = spawned_state.pending_police_bank_indices.at[i].set(-1)
                new_pending_scores = spawned_state.pending_police_scores.at[i].set(-1)
                new_pending_positions = spawned_state.pending_police_spawn_positions.at[i].set(jnp.array([-1, -1]))

                return spawned_state.replace(
                    pending_police_spawns=new_pending_spawns,
                    pending_police_bank_indices=new_pending_indices,
                    pending_police_spawn_positions=new_pending_positions,
                    pending_police_scores=new_pending_scores
                )
            
            ready_to_spawn = current_state.pending_police_spawns[i] == 0
            return jax.lax.cond(ready_to_spawn, spawn_at_bank_index, lambda s: s, current_state)
        
        return jax.lax.fori_loop(0, len(state.pending_police_spawns), process_single_spawn, state)
    
    @partial(jax.jit, static_argnums=(0,))
    def place_dynamite(self, state: BankHeistState, player: Entity) -> BankHeistState:
        dynamite_inactive = jnp.all(state.dynamite_position == jnp.array([-1, -1]))
        
        def place_new_dynamite():
            player_position = player.position
            player_direction = player.direction
            
            offset_branches = [
                lambda: jnp.array([0, -5]),   # DOWN -> place 4 pixels UP (behind)
                lambda: jnp.array([0, 5]),    # UP -> place 4 pixels DOWN (behind)
                lambda: jnp.array([-5, 2]),   # RIGHT -> place 4 pixels LEFT (behind)
                lambda: jnp.array([5, 2]),    # LEFT -> place 4 pixels RIGHT (behind)
                lambda: jnp.array([0, 0]),    # NOOP/FIRE -> place at current position
            ]
            offset = jax.lax.switch(player_direction, offset_branches)
            
            new_dynamite_position = player_position + offset
            new_dynamite_timer = jnp.array([self.consts.DYNAMITE_EXPLOSION_DELAY]).astype(jnp.int32)
            return state.replace(
                dynamite_position=new_dynamite_position,
                dynamite_timer=new_dynamite_timer,
                fuel=state.fuel - self.consts.DYNAMITE_COST
            )
        return jax.lax.cond(dynamite_inactive, place_new_dynamite, lambda: state)

    @partial(jax.jit, static_argnums=(0,))
    def kill_police_in_explosion_range(self, state: BankHeistState) -> BankHeistState:
        dynamite_pos = state.dynamite_position
        police_killed_count = jnp.array(0)
        total_score_added = jnp.array(0)
        
        def check_and_kill_police(i, carry):
            current_state, killed_count, score_added = carry
            police_pos = current_state.enemy_positions.position[i]
            police_visible = current_state.enemy_positions.visibility[i] > 0
            
            distance = jnp.linalg.norm(dynamite_pos - police_pos)
            within_range = distance <= 5
            should_kill = police_visible & within_range
            
            inactive_count = jnp.sum(current_state.enemy_positions.visibility == 0)
            kill_reward = jnp.where(inactive_count == 0, self.consts.POLICE_KILL_REWARD[0],
                         jnp.where(inactive_count == 1, self.consts.POLICE_KILL_REWARD[1], self.consts.POLICE_KILL_REWARD[2]))
            
            new_visibility = current_state.enemy_positions.visibility.at[i].set(
                jnp.where(should_kill, 0, current_state.enemy_positions.visibility[i])
            )
            new_respawn_timer = current_state.bank_spawn_timers.at[i].set(
                jnp.where(should_kill, self.consts.BANK_RESPAWN_TIME, current_state.bank_spawn_timers[i])
            )

            new_police = current_state.enemy_positions.replace(visibility=new_visibility)
            
            # MERGED: Store the killed police score UI
            new_killed_score = current_state.killed_police_scores.at[i].set(
                jnp.where(should_kill, kill_reward // 10, current_state.killed_police_scores[i])
            )
            
            updated_state = current_state.replace(
                enemy_positions=new_police, 
                bank_spawn_timers=new_respawn_timer,
                killed_police_scores=new_killed_score
            )
            
            new_killed_count = killed_count + jnp.where(should_kill, 1, 0)
            new_score_added = score_added + jnp.where(should_kill, kill_reward, 0)
            
            return (updated_state, new_killed_count, new_score_added)
        
        final_state, total_killed, total_score = jax.lax.fori_loop(
            0, len(state.enemy_positions.visibility), 
            check_and_kill_police, 
            (state, police_killed_count, total_score_added)
        )
        
        new_money = final_state.money + total_score
        return final_state.replace(money=new_money)
    
    @partial(jax.jit, static_argnums=(0,))
    def check_player_in_explosion_range(self, state: BankHeistState) -> chex.Array:
        dynamite_pos = state.dynamite_position
        player_pos = state.player.position
        distance = jnp.linalg.norm(dynamite_pos - player_pos)
        return distance <= 5
    
    @partial(jax.jit, static_argnums=(0,))
    def explode_dynamite(self, state: BankHeistState) -> BankHeistState:
        updated_state = self.kill_police_in_explosion_range(state)
        
        player_in_range = self.check_player_in_explosion_range(updated_state)
        updated_state = jax.lax.cond(player_in_range, lambda: self.lose_life(updated_state), lambda: updated_state)
        
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        
        # MERGED: Add visual flash logic for explosions
        new_explosion_timer = jnp.array(60).astype(jnp.int32) 
        
        return updated_state.replace(
            dynamite_position=new_dynamite_position,
            dynamite_timer=new_dynamite_timer,
            explosion_timer=new_explosion_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: BankHeistState) -> BankHeistState:
        # Preserve per-city snapshots for mods that revisit or randomize cities.
        saved_city_states = self.save_city_state(
            state.city_states,
            state.map_id,
            state.bank_positions,
            state.bank_spawn_timers,
            state.bank_spawn_indices,
            state.bank_heists,
        )

        new_level = state.level + 1
        new_difficulty_level = jnp.minimum(new_level // self.consts.CITIES_PER_LEVEL, self.consts.MAX_LEVEL - 1)
        num_maps = len(self.city_collision_maps)
        is_new_loop = (new_level % num_maps == 0) & (new_level > state.level)
        saved_city_states = jax.lax.cond(
            is_new_loop,
            lambda cs: self._init_city_states(),
            lambda cs: cs,
            saved_city_states,
        )
        new_map_id = new_level % num_maps

        # ALE spawns 1px further right after level transitions (but not on initial reset).
        default_player_position = jnp.array(self.consts.LEVEL_TRANSITION_SPAWN).astype(jnp.int32)
        new_player = state.player.replace(position=default_player_position)
        empty_police = init_banks_or_police()
        empty_banks = init_banks_or_police()
        
        new_speed = self.consts.BASE_SPEED * jnp.power(self.consts.SPEED_INCREASE_PER_LEVEL, new_difficulty_level)
        new_fuel_consumption = jnp.power(self.consts.FUEL_CONSUMPTION_INCREASE_PER_LEVEL, new_difficulty_level)
        
        # Fuel refill: if they robbed 0 banks, REFILL_TABLE[0] is 0 and they keep current fuel.
        new_fuel = jnp.maximum(state.fuel, jax.lax.dynamic_index_in_dim(self.consts.REFILL_TABLE, state.bank_heists, axis=0, keepdims=False))
        new_fuel_refill = jnp.array(0).astype(jnp.int32)
        
        new_map_collision = jax.lax.dynamic_index_in_dim(self.city_collision_maps, new_map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(self.city_spawns, new_map_id, axis=0, keepdims=False)
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)
        new_bank_spawn_timers = jnp.array([1, 1, 1]).astype(jnp.int32)
        new_police_spawn_timers = jnp.array([-1, -1, -1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        new_explosion_timer = jnp.array(0).astype(jnp.int32)
        
        new_player_lives = jax.lax.cond(
            state.bank_heists >= 9, 
            lambda: state.player_lives + 1, 
            lambda: state.player_lives
        )
        
        city_reward_index = state.level % 4
        city_reward = jnp.where(city_reward_index == 0, self.consts.CITY_REWARD[0],
                      jnp.where(city_reward_index == 1, self.consts.CITY_REWARD[1],
                      jnp.where(city_reward_index == 2, self.consts.CITY_REWARD[2], self.consts.CITY_REWARD[3])))
        
        total_bonus = jax.lax.cond(
            state.bank_heists >= 9,
            lambda: city_reward + (state.difficulty_level * self.consts.BONUS_REWARD),
            lambda: jnp.array(0).astype(jnp.int32)
        )
        new_money = state.money + total_bonus
        
        return state.replace(
            level=new_level,
            map_id=new_map_id,
            difficulty_level=new_difficulty_level,
            player=new_player,
            player_move_direction=new_player.direction,
            portal_pending=jnp.array(False).astype(jnp.bool_),
            portal_pending_side=jnp.array(-1).astype(jnp.int32),
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
            explosion_timer=new_explosion_timer,
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_scores=jnp.array([-1, -1, -1]).astype(jnp.int32),
            killed_police_scores=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_spawn_positions=jnp.full((3, 2), -1).astype(jnp.int32),
            bank_heists=jnp.array(0).astype(jnp.int32),
            # Keep ALE-style global bank sequence across city transitions.
            bank_spawn_indices=state.bank_spawn_indices,
            city_states=saved_city_states,
            pending_exit=jnp.array(False).astype(jnp.bool_),
            random_key=jax.random.fold_in(state.random_key, new_level + 100)
        )

    @partial(jax.jit, static_argnums=(0,))
    def move(self, position: Entity, direction: int) -> Entity:
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
        temp_entity = Entity(position=position,direction=jnp.array(direction), visibility=jnp.array(1))
        new_position = self.move(temp_entity, direction)
        collision = self.check_background_collision(state, new_position)
        return collision < 255

    @partial(jax.jit, static_argnums=(0,))
    def get_valid_directions(self, state: BankHeistState, position: chex.Array, current_direction: int) -> chex.Array:
        all_directions = jnp.array([self.consts.DIR_DOWN, self.consts.DIR_UP, self.consts.DIR_RIGHT, self.consts.DIR_LEFT])
        
        reverse_direction = jax.lax.switch(current_direction, [
            lambda: self.consts.DIR_UP,   
            lambda: self.consts.DIR_DOWN, 
            lambda: self.consts.DIR_LEFT, 
            lambda: self.consts.DIR_RIGHT,
            lambda: -1,
        ])
        
        def check_direction(direction):
            is_reverse = direction == reverse_direction
            is_valid = self.check_valid_direction(state, position, direction)
            return is_valid & (~is_reverse)
        
        valid_mask = jax.vmap(check_direction)(all_directions)
        valid_directions = jnp.where(valid_mask, all_directions, -1)
        return valid_directions

    @partial(jax.jit, static_argnums=(0,))
    def choose_police_direction(self, state: BankHeistState, police_position: chex.Array, current_direction: int, random_key: chex.PRNGKey) -> int:
        valid_directions = self.get_valid_directions(state, police_position, current_direction)
        player_position = state.player.position
        
        reverse_direction = jax.lax.switch(current_direction, [
            lambda: self.consts.DIR_UP,
            lambda: self.consts.DIR_DOWN,
            lambda: self.consts.DIR_LEFT,
            lambda: self.consts.DIR_RIGHT,
            lambda: current_direction,
        ])

        valid_count = jnp.sum(valid_directions >= 0)

        # If no valid directions, allow reversing to escape dead ends
        def no_valid_directions():
            can_reverse = self.check_valid_direction(state, police_position, reverse_direction)
            return jax.lax.cond(
                can_reverse,
                lambda: reverse_direction,
                lambda: current_direction
            )
        
        def one_valid_direction():
            valid_mask = valid_directions >= 0
            return jnp.where(valid_mask, valid_directions, 0).max()
        
        def multiple_valid_directions():
            def get_new_position(direction):
                temp_entity = Entity(position=police_position, direction=jnp.array(direction), visibility=jnp.array(1))
                moved_entity = self.move(temp_entity, direction)
                return moved_entity.position
            
            new_positions = jax.vmap(get_new_position)(jnp.array([self.consts.DIR_DOWN, self.consts.DIR_UP, self.consts.DIR_RIGHT, self.consts.DIR_LEFT]))
            distances = jnp.linalg.norm(new_positions - player_position[None, :], axis=1)
            
            distance_bias = -distances
            distance_bias = distance_bias - jnp.min(distance_bias) 
            max_bias = jnp.max(distance_bias)
            distance_bias = jnp.where(max_bias > 0, distance_bias / max_bias, 0.0)
            
            base_weights = jnp.where(valid_directions >= 0, 1.0, 0.0)
            
            random_noise = jax.random.uniform(random_key, shape=(4,))
            combined_weights = base_weights * (
                self.consts.POLICE_RANDOM_FACTOR * random_noise + 
                self.consts.POLICE_BIAS_FACTOR * (distance_bias + 0.1) 
            )
            
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
        def move_single_police(i, current_state):
            def move_police_car(state_inner):
                police_position = state_inner.enemy_positions.position[i]
                current_direction = state_inner.enemy_positions.direction[i]
                
                police_key = jax.random.fold_in(random_key, i)
                new_direction = self.choose_police_direction(state_inner, police_position, current_direction, police_key)
                
                temp_entity = Entity(
                    position=police_position,
                    direction=jnp.array(new_direction),
                    visibility=jnp.array(1)
                )
                moved_entity = self.move(temp_entity, new_direction)
                collision = self.check_background_collision(state_inner, moved_entity)
                
                moved_entity = jax.lax.cond(collision >= 255,
                    lambda: temp_entity, 
                    lambda: moved_entity 
                )
                
                moved_entity = self.portal_handler(moved_entity, collision, True)
                
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
        step_random_key, next_random_key  = jax.random.split(state.random_key)
        updated_state = state.replace(random_key=next_random_key)
        return step_random_key, updated_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        step_random_key, new_state = self.advance_random_key(state)

        # Determine the new priority if the action changed to a diagonal
        action_changed = atari_action != state.latched_action
        new_priority = jnp.where(
            action_changed,
            jnp.select(
                condlist=[
                    atari_action == Action.UPRIGHT,
                    atari_action == Action.UPLEFT,
                    atari_action == Action.DOWNRIGHT,
                    atari_action == Action.DOWNLEFT,
                    atari_action == Action.UPRIGHTFIRE,
                    atari_action == Action.UPLEFTFIRE,
                    atari_action == Action.DOWNRIGHTFIRE,
                    atari_action == Action.DOWNLEFTFIRE,
                ],
                choicelist=[
                    jnp.where(state.latched_action == Action.UP, self.consts.DIR_RIGHT, self.consts.DIR_UP),
                    jnp.where(state.latched_action == Action.UP, self.consts.DIR_LEFT, self.consts.DIR_UP),
                    jnp.where(state.latched_action == Action.DOWN, self.consts.DIR_RIGHT, self.consts.DIR_DOWN),
                    jnp.where(state.latched_action == Action.DOWN, self.consts.DIR_LEFT, self.consts.DIR_DOWN),
                    jnp.where(jnp.logical_or(state.latched_action == Action.UP, state.latched_action == Action.UPFIRE), self.consts.DIR_RIGHT, self.consts.DIR_UP),
                    jnp.where(jnp.logical_or(state.latched_action == Action.UP, state.latched_action == Action.UPFIRE), self.consts.DIR_LEFT, self.consts.DIR_UP),
                    jnp.where(jnp.logical_or(state.latched_action == Action.DOWN, state.latched_action == Action.DOWNFIRE), self.consts.DIR_RIGHT, self.consts.DIR_DOWN),
                    jnp.where(jnp.logical_or(state.latched_action == Action.DOWN, state.latched_action == Action.DOWNFIRE), self.consts.DIR_LEFT, self.consts.DIR_DOWN),
                ],
                default=state.current_diagonal_priority
            ),
            state.current_diagonal_priority
        )

        # Update the state with the newly calculated priority
        new_state = new_state.replace(current_diagonal_priority=new_priority)

        # Apply player input EVERY frame (even when movement doesn't happen),
        # but with a 1-frame latch to match ALE timing.
        new_state = self.player_input_step(new_state, state.latched_action)

        # Movement is gated by the speed accumulator (fractional speeds move every N frames).
        full_speed = new_state.speed + new_state.reserve_speed
        movement_ticks = full_speed.astype(jnp.int32)

        # ALE phase: at speeds < 1 pixel/frame, the movement tick is only consumed
        # on one of the two frame phases.
        is_subpixel = new_state.speed < 1.0
        consume_phase = (state.time % 2) == 0  # allow consumption on even times
        movement_ticks = jnp.where(jnp.logical_and(is_subpixel, jnp.logical_not(consume_phase)), 0, movement_ticks)

        # Update reserve using "used distance" (not mod) so we can carry >1.0 when phase-blocked.
        new_reserve = full_speed - movement_ticks.astype(jnp.float32)
        new_state = new_state.replace(reserve_speed=new_reserve)
        new_state = jax.lax.fori_loop(0, movement_ticks, lambda i, s: self.player_move_step(s), new_state)

        # Latch the *current* action for next frame (ALE behavior).
        new_state = new_state.replace(latched_action=atari_action)
        police_speed = jax.lax.cond(
            new_state.game_paused,
            lambda: jnp.array(0).astype(jnp.float32),
            lambda: (new_state.speed * self.consts.POLICE_SLOW_DOWN_FACTOR + state.police_reserve_speed)        )
        new_state = new_state.replace(police_reserve_speed=jnp.mod(police_speed, 1.0))
        police_speed = police_speed.astype(jnp.int32)
        new_state = jax.lax.fori_loop(0, police_speed, lambda i, s: self.move_police_cars(s, step_random_key), new_state)

        new_state = jax.lax.cond(
            new_state.game_paused,
            lambda: new_state,
            lambda: self.timer_step(new_state,step_random_key)
        )

        new_state = jax.lax.cond(
            new_state.game_paused,
            lambda: new_state,
            lambda: self.fuel_step(new_state)
        )

        new_state = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_not(new_state.game_paused),
                jnp.logical_and(new_state.player_lives >= 0, new_state.pending_exit),
            ),
            lambda: self.map_transition(new_state),
            lambda: new_state,
        )

        done = new_state.player_lives < 0
        obs = self._get_observation(new_state)
        
        if new_state.obs_stack is not None:
            old_stack = new_state.obs_stack
            def shift_and_add(x_old, x_new):
                shifted = x_old[1:] 
                x_new_expanded = jnp.expand_dims(x_new, axis=0)
                return jnp.concatenate([shifted, x_new_expanded], axis=0)
            
            new_obs_stack = jax.tree.map(shift_and_add, old_stack, obs)
            new_state = new_state.replace(obs_stack=new_obs_stack)
            
        reward = new_state.money - state.money
        new_time = (state.time + 1).astype(jnp.int32)
        new_state = new_state.replace(time=new_time)

        info = BankHeistInfo(
            time=new_state.time,
        )

        return self._get_observation(new_state), new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def player_input_step(self, state: BankHeistState, atari_action: chex.Array) -> BankHeistState:
        internal_direction = jnp.select(
            condlist=[
                atari_action == Action.UPFIRE,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.UP,
                atari_action == Action.DOWN,
                atari_action == Action.LEFT,
                atari_action == Action.RIGHT,
                atari_action == Action.UPRIGHT,
                atari_action == Action.UPLEFT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.DOWNLEFT,
            ],
            choicelist=[
                self.consts.DIR_UP,     
                self.consts.DIR_DOWN,   
                self.consts.DIR_LEFT,   
                self.consts.DIR_RIGHT,  
                self.consts.DIR_UP,     
                self.consts.DIR_UP,     
                self.consts.DIR_DOWN,   
                self.consts.DIR_DOWN,   
                self.consts.DIR_UP,     
                self.consts.DIR_DOWN,   
                self.consts.DIR_LEFT,   
                self.consts.DIR_RIGHT,  
                self.consts.DIR_UP,     
                self.consts.DIR_UP,     
                self.consts.DIR_DOWN,   
                self.consts.DIR_DOWN,   
            ],
            default=self.consts.DIR_NOOP, 
        )
        
        player_input = jnp.where(
            jnp.logical_or(atari_action == Action.NOOP, atari_action == Action.FIRE),
            state.player_move_direction,
            internal_direction,
        )

        faced_player = state.player.replace(direction=player_input)

        base_player = state.player.replace(direction=state.player_move_direction)
        validated_player, invalid_input = self.validate_input(state, base_player, player_input)
        faced_player = faced_player.replace(direction=validated_player.direction)

        new_state = state.replace(
            player=faced_player,
            player_move_direction=state.player_move_direction,
            game_paused=jnp.where(
                jnp.logical_and(jnp.logical_not(invalid_input), atari_action != Action.NOOP),
                jnp.array(False).astype(jnp.bool_),
                state.game_paused,
            ),
        )

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
            lambda: self.place_dynamite(new_state, new_state.player),
            lambda: new_state
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def player_move_step(self, state: BankHeistState) -> BankHeistState:
        def do_move(moving_state: BankHeistState) -> BankHeistState:
            current_player = moving_state.player

            desired_dir = current_player.direction
            base_player = current_player.replace(direction=moving_state.player_move_direction)
            turning_player, _ = self.validate_input(moving_state, base_player, desired_dir)
            new_move_dir = turning_player.direction

            attempted_player = self.move(current_player, new_move_dir)
            collision = self.check_background_collision(moving_state, attempted_player)
            new_player = jax.lax.cond(
                collision >= 255,
                lambda: current_player,
                lambda: attempted_player,
            )

            new_player = new_player.replace(direction=new_move_dir)

            moved_state = moving_state.replace(
                player=new_player,
                player_move_direction=new_move_dir,
            )

            side = (new_player.position[0] <= 80).astype(jnp.int32)
            should_teleport_now = jnp.logical_and(moved_state.portal_pending, collision == 100)

            def teleport(s: BankHeistState) -> BankHeistState:
                teleported_player = s.player.replace(
                    position=jnp.array([self.consts.PORTAL_X[s.portal_pending_side], s.player.position[1]])
                )
                return s.replace(
                    player=teleported_player,
                    player_move_direction=teleported_player.direction,
                    portal_pending=jnp.array(False).astype(jnp.bool_),
                    portal_pending_side=jnp.array(-1).astype(jnp.int32),
                )

            moved_state = jax.lax.cond(should_teleport_now, teleport, lambda s: s, moved_state)

            moved_state = moved_state.replace(
                portal_pending=jnp.where(should_teleport_now, jnp.array(False).astype(jnp.bool_), collision == 100),
                portal_pending_side=jnp.where(
                    should_teleport_now,
                    jnp.array(-1).astype(jnp.int32),
                    jnp.where(collision == 100, side, jnp.array(-1).astype(jnp.int32)),
                ),
            )

            _, bank_hit_index = self.check_bank_collision(new_player, moved_state.bank_positions)
            bank_hit = bank_hit_index >= 0
            moved_state = jax.lax.cond(bank_hit, lambda: self.handle_bank_robbery(moved_state, bank_hit_index), lambda: moved_state)

            _, police_hit_index = self.check_police_collision(new_player, moved_state.enemy_positions)
            police_hit = police_hit_index >= 0
            moved_state = jax.lax.cond(police_hit, lambda: self.lose_life(moved_state), lambda: moved_state)

            moved_state = moved_state.replace(
                pending_exit=jnp.logical_or(moved_state.pending_exit, collision == 200)
            )

            moved_state = jax.lax.cond(
                jnp.logical_and(moved_state.game_paused, jnp.logical_not(police_hit)),
                lambda s: moving_state,
                lambda s: moved_state,
                operand=None,
            )
            return moved_state

        return jax.lax.cond(state.game_paused, lambda s: state, do_move, state)
    
    @partial(jax.jit, static_argnums=(0,))
    def fuel_step(self, state: BankHeistState) -> BankHeistState:
        new_fuel = jnp.maximum(state.fuel - state.fuel_consumption, 0)
        new_state = state.replace(fuel=new_fuel)
        new_state = jax.lax.cond(
            new_state.fuel <= 0,
            lambda: self.lose_life(new_state).replace(fuel=self.consts.REVIVAL_FUEL),
            lambda: new_state
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def spawn_banks_fn(self, state: BankHeistState, step_random_key: chex.PRNGKey) -> BankHeistState:
        """Deterministic ALE behavior: each bank independently traverses the 16-step array."""
        del step_random_key  # Unused in deterministic behavior

        spawning_mask = state.bank_spawn_timers == 0  # (3,)
        chosen_points = self.consts.HARDCODED_BANK_SPAWNS[state.bank_spawn_indices]  # (3,2)

        new_indices = jnp.where(
            spawning_mask,
            (state.bank_spawn_indices + 1) % 16,
            state.bank_spawn_indices,
        )

        new_bank_positions = jnp.where(
            spawning_mask[:, None], chosen_points, state.bank_positions.position
        )
        new_visibility = jnp.where(
            state.bank_spawn_timers == 0,
            jnp.array([1, 1, 1]),
            state.bank_positions.visibility,
        )

        new_banks = state.bank_positions.replace(
            position=new_bank_positions, visibility=new_visibility
        )
        return state.replace(bank_positions=new_banks, bank_spawn_indices=new_indices)

    @partial(jax.jit, static_argnums=(0,))
    def timer_step(self, state: BankHeistState, step_random_key: chex.PRNGKey) -> BankHeistState:
        new_bank_spawn_timers = jnp.where(state.bank_spawn_timers >= 0, state.bank_spawn_timers - 1, state.bank_spawn_timers)
        new_police_spawn_timers = jnp.where(state.police_spawn_timers >= 0, state.police_spawn_timers - 1, state.police_spawn_timers)
        new_dynamite_timer = jnp.where(state.dynamite_timer >= 0, state.dynamite_timer - 1, state.dynamite_timer)
        new_explosion_timer = jnp.where(state.explosion_timer > 0, state.explosion_timer - 1, state.explosion_timer) # MERGED

        new_pending_police_spawns = jnp.where(state.pending_police_spawns >= 0, state.pending_police_spawns - 1, state.pending_police_spawns)

        new_state = state.replace(
            bank_spawn_timers=new_bank_spawn_timers,
            police_spawn_timers=new_police_spawn_timers,
            dynamite_timer=new_dynamite_timer,
            explosion_timer=new_explosion_timer,  # MERGED
            pending_police_spawns=new_pending_police_spawns
        )
        # Spawn banks when their timers reach 0
        spawn_bank_condition = jnp.any(new_bank_spawn_timers == 0)
        new_state = jax.lax.cond(
            spawn_bank_condition,
            lambda: self.spawn_banks_fn(new_state, step_random_key),
            lambda: new_state,
        )
        
        spawn_police_condition = jnp.any(new_pending_police_spawns == 0)
        new_state = jax.lax.cond(
            spawn_police_condition,
            lambda: self.process_pending_police_spawns(new_state),
            lambda: new_state,
        )
        
        dynamite_explode_condition = state.dynamite_timer[0] == 0
        new_state = jax.lax.cond(
            dynamite_explode_condition,
            lambda: self.explode_dynamite(new_state),
            lambda: new_state,
        )
        
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
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        sprite_path = self.consts.SPRITES_DIR
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        city_asset = next((a for a in final_asset_config if a['name'] == 'cities'), None)
        if city_asset:
            final_asset_config.remove(city_asset)
            city_files = city_asset['files']
            final_asset_config.append({'name': 'background', 'type': 'background', 'file': city_files[0]})
            final_asset_config.append({'name': 'city_maps', 'type': 'group', 'files': city_files})
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # --- Manual Padding for Cars ---
        p_side_mask = self.SHAPE_MASKS['player_side']   
        p_front_mask = self.SHAPE_MASKS['player_front'] 
        
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
        
        # --- Procedural color IDs ---
        self.fuel_color_id = self.COLOR_TO_ID[(167, 26, 26)]
        self.black_color_id = self.COLOR_TO_ID.get((0, 0, 0), 0)

    @partial(jax.jit, static_argnums=(0,))
    def _render_fuel_tank(self, raster, state):
        tank_mask = self.SHAPE_MASKS['fuel_tank'] 
        tank_x, tank_y = self.consts.FUEL_TANK_POSITION
        raster = self.jr.render_at(raster, tank_x, tank_y, tank_mask, flip_offset=self.FLIP_OFFSETS['fuel_tank'])
        level = state.fuel / self.consts.FUEL_CAPACITY
        
        scaled_tank_x = jnp.round(tank_x * self.jr.config.width_scaling).astype(jnp.int32)
        scaled_tank_y = jnp.round(tank_y * self.jr.config.height_scaling).astype(jnp.int32)
        
        sprite_h, sprite_w = tank_mask.shape 
        
        pixel_level = jnp.ceil(level * sprite_h).astype(jnp.int32)
        
        xx, yy = self.jr._xx, self.jr._yy
        
        fill_top_y = scaled_tank_y + sprite_h - pixel_level
        
        fill_box_mask = (yy >= fill_top_y) & (yy < scaled_tank_y + sprite_h) & \
                        (xx >= scaled_tank_x) & (xx < scaled_tank_x + sprite_w)
        
        sprite_coords_y = yy - scaled_tank_y
        sprite_coords_x = xx - scaled_tank_x
        sampled_tank_mask = map_coordinates(
            tank_mask,
            [sprite_coords_y, sprite_coords_x],
            order=0,
            cval=self.jr.TRANSPARENT_ID
        )
        tank_pixels_mask = (sampled_tank_mask != self.jr.TRANSPARENT_ID)
        final_fill_mask = fill_box_mask & tank_pixels_mask
        
        return jnp.where(final_fill_mask, jnp.asarray(self.fuel_color_id, raster.dtype), raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_lives(self, raster, state, life_mask):
        def body_fun(i, rast):
            return jax.lax.cond(
                i < state.player_lives,
                lambda r: self.jr.render_at(r, self.consts.LIFE_POSITIONS[i][0], self.consts.LIFE_POSITIONS[i][1], life_mask, flip_offset=jnp.array([0,0])),
                lambda r: r,
                rast
            )
        raster = jax.lax.fori_loop(0, len(self.consts.LIFE_POSITIONS), body_fun, raster)
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        ### Render City
        city_mask = self.SHAPE_MASKS['city_maps'][state.map_id % self.SHAPE_MASKS['city_maps'].shape[0]]
        raster = self.jr.render_at(raster, 0, 0, city_mask, flip_offset=self.FLIP_OFFSETS['city_maps'])
        ### Render Player
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
        raster = self.jr.render_at(raster, state.player.position[0], state.player.position[1], player_mask, flip_offset=jnp.array([0,0]))
        
        ### Render Fuel Tank
        raster = self._render_fuel_tank(raster, state)
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
        life_mask = self.PLAYER_MASKS[0] 
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

        ### MERGED: Render Pending Bank Scores (Uses exact spawn positions)
        def render_bank_score(i, r):
            score_index = state.pending_police_scores[i] - 1
            safe_score_index = jnp.clip(score_index, 0, 8)
            score_mask = self.SHAPE_MASKS['bank_scores'][safe_score_index]
            score_flip_offset = self.FLIP_OFFSETS['bank_scores']
            
            # Using File 2's specific spawn_positions array instead of bank array
            safe_x = state.pending_police_spawn_positions[i, 0]
            safe_y = state.pending_police_spawn_positions[i, 1]
            
            return jax.lax.cond(
                state.pending_police_spawns[i] > 0,
                lambda r_in: self.jr.render_at(r_in, safe_x, safe_y, score_mask, flip_offset=score_flip_offset),
                lambda r_in: r_in,
                r
            )
        raster = jax.lax.fori_loop(0, state.pending_police_spawns.shape[0], render_bank_score, raster)
        
        ### MERGED: Render Killed Police Scores
        def render_killed_police_score(i, r):
            score_index = state.killed_police_scores[i] - 1
            safe_score_index = jnp.clip(score_index, 0, 8)
            score_mask = self.SHAPE_MASKS['bank_scores'][safe_score_index]
            score_flip_offset = self.FLIP_OFFSETS['bank_scores']
            
            display_condition = state.bank_spawn_timers[i] > (self.consts.BANK_RESPAWN_TIME - 120)
            
            return jax.lax.cond(
                display_condition,
                lambda r_in: self.jr.render_at(r_in, state.enemy_positions.position[i, 0], state.enemy_positions.position[i, 1], score_mask, flip_offset=score_flip_offset),
                lambda r_in: r_in,
                r
            )
        raster = jax.lax.fori_loop(0, state.bank_spawn_timers.shape[0], render_killed_police_score, raster)

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
                return self.jr.render_at(raster_input, state.enemy_positions.position[i, 0], state.enemy_positions.position[i, 1], police_mask, flip_offset=jnp.array([0,0]))
            
            return jax.lax.cond(
                state.enemy_positions.visibility[i] != 0,
                render_police,
                lambda r_in: r_in,
                r
            )
        raster = jax.lax.fori_loop(0, state.enemy_positions.position.shape[0], render_police_car, raster)
        
        ### MERGED: Render Dynamite (with blink effect)
        def render_dynamite(raster_input):
            dynamite_mask = self.SHAPE_MASKS['dynamite']
            return self.jr.render_at(raster_input, state.dynamite_position[0], state.dynamite_position[1], dynamite_mask, flip_offset=self.FLIP_OFFSETS['dynamite'])
        
        dynamite_active = ~jnp.all(state.dynamite_position == jnp.array([-1, -1]))
        dynamite_visible = dynamite_active & ((state.dynamite_timer[0] % 8) < 4)
        raster = jax.lax.cond(dynamite_visible, render_dynamite, lambda r: r, raster)
        
        ### Render Score
        score_digits_arr = self.jr.int_to_digits(state.money, max_digits=4)
        raster = self.jr.render_label_selective(
            raster, 90, 179, 
            all_digits=score_digits_arr, 
            digit_id_masks=self.SHAPE_MASKS['digits'], 
            start_index=0, 
            num_to_render=4,
            spacing=12,
            max_digits_to_render=4
        )

        ### MERGED: Render Explosion Flashes
        def apply_flash(palette):
            flash_idx = state.explosion_timer % 15
            is_light_grey = (flash_idx >= 5) & (flash_idx < 10)
            flash_color = jnp.where(
                is_light_grey,
                jnp.array([170, 170, 170], dtype=jnp.uint8),
                jnp.array([100, 100, 100], dtype=jnp.uint8)
            )
            return palette.at[self.black_color_id].set(flash_color)

        final_palette = jax.lax.cond(
            state.explosion_timer > 0,
            apply_flash,
            lambda p: p,
            self.PALETTE
        )

        return self.jr.render_from_palette(raster, final_palette)