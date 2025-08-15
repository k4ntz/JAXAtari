import os
from functools import partial
from typing import NamedTuple, Tuple
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

COLLISION_BOX =(8,8)
PORTAL_X = jnp.array([12, 140])

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(MODULE_DIR, "sprites", "bankheist")

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

CITY_COLLISION_MAPS = [load_city_collision_map(f"map_{i+1}_collision.npy") for i in range(8)]
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
    enemy_positions: chex.Array
    bank_positions: chex.Array
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
            enemy_positions=jnp.array([]).astype(jnp.int32),
            bank_positions=jnp.array([None,None,None]).astype(jnp.int32),
            speed=jnp.array(1).astype(jnp.int32),
            money=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(4).astype(jnp.int32),
            fuel_refill=jnp.array(0).astype(jnp.int32),
            obs_stack=None,
            map_collision=CITY_COLLISION_MAPS[0],
            spawn_points=CITY_SPAWNS[0],
            bank_spawn_timers=jnp.array([1]).astype(jnp.int32),
            police_spawn_timers=jnp.array([0]).astype(jnp.int32),
            dynamite_timer=jnp.array([0]).astype(jnp.int32)
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
    def map_transition(self, state: BankHeistState) -> BankHeistState:

        new_level = state.level+1
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player._replace(position=default_player_position)
        empty_police = jnp.array([None,None,None]).astype(jnp.int32)
        empty_banks = jnp.array([None,None,None]).astype(jnp.int32)
        new_speed = state.speed * 1.1
        new_fuel = state.fuel_refill
        new_fuel_refill=jnp.array(0).astype(jnp.int32)
        collision_branches = [lambda: map for map in CITY_COLLISION_MAPS]
        spawn_branches = [lambda: points for points in CITY_SPAWNS]
        map_id = new_level % len(CITY_COLLISION_MAPS)
        new_map_collision = jax.lax.switch(map_id, collision_branches)
        new_spawn_points = jax.lax.switch(map_id, spawn_branches)
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
            dynamite_timer=new_dynamite_timer
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
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        # Player step
        new_state = self.player_step(state, action)
        # Timer step
        #new_state = self.timer_step(new_state)
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

        #new_state = jax.lax.cond(collision == 200, lambda: self.map_transition(new_state), lambda: new_state)
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
            new_bank_spawns = jax.random.randint(key, shape=state.bank_positions.shape, minval=0, maxval=state.spawn_points.shape[0])
            new_pos = Entity(new_bank_spawns[0], 0, 1)
            new_bank_positions = jnp.where(state.bank_positions == 0, new_pos, state.bank_positions)
            return state._replace(bank_positions=new_bank_positions)

        new_bank_spawn_timers = jnp.where(state.bank_spawn_timers >= 0, state.bank_spawn_timers - 1, state.bank_spawn_timers)
        new_police_spawn_timers = jnp.where(state.police_spawn_timers >= 0, state.police_spawn_timers - 1, state.police_spawn_timers)
        new_dynamite_timer = jnp.where(state.dynamite_timer >= 0, state.dynamite_timer - 1, state.dynamite_timer)

        new_state = state._replace(
            bank_spawn_timers=new_bank_spawn_timers,
            police_spawn_timers=new_police_spawn_timers,
            dynamite_timer=new_dynamite_timer
        )
        bank_update = spawn_bank(new_state)
        new_state = jnp.where(new_bank_spawn_timers == 0, bank_update, new_state)
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
        
        # Render City
        frame_city = aj.get_sprite_frame(self.SPRITES_CITY[state.level], 0)
        raster = aj.render_at(raster, 0, 0, frame_city)

        # Render Player
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