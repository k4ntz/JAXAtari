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

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    direction: jnp.ndarray

class BankHeistState(NamedTuple):
    level: chex.Array
    player_position: chex.Array
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

#TODO: Add Background collision Map, Fuel, Fuel Refill and others
class BankHeistObservation(NamedTuple):
    player: EntityPosition
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
            fuel = jnp.array(100).astype(jnp.int32),
            player_position=EntityPosition(
                x=jnp.array(12).astype(jnp.int32),
                y=jnp.array(78).astype(jnp.int32),
                width=jnp.array(16).astype(jnp.int32),
                height=jnp.array(16).astype(jnp.int32),
                direction=jnp.array(4).astype(jnp.int32)
            ),
            dynamite_position=jnp.array([]).astype(jnp.int32),
            enemy_positions=jnp.array([]).astype(jnp.int32),
            bank_positions=jnp.array([]).astype(jnp.int32),
            speed=jnp.array(1).astype(jnp.int32),
            money=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(4).astype(jnp.int32),
            fuel_refill=jnp.array(0).astype(jnp.int32),
            obs_stack=None,
            map_collision=load_city_collision_map("map_1_collision.npy"),
        )
        obs = self._get_observation(state)
        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
        obs_stack = jax.tree.map(expand_and_copy, obs)
        state = state._replace(obs_stack=obs_stack)
        return  obs_stack, state

    
    @partial(jax.jit, static_argnums=(0,))
    def validate_input(self, state: BankHeistState, position: EntityPosition, input: jnp.ndarray) -> EntityPosition:
        """
        Confirm that the player is not trying to move into a wall.

        Returns:
            EntityPosition: Contains the new direction of the player after validating the input.
        """
        new_position = self.move(position, input, state.speed)
        new_position = new_position._replace(direction=input)
        collision = self.check_background_collision(state, new_position)
        direction = jax.lax.cond(collision >= 255,
            lambda: position.direction,
            lambda: new_position.direction
        )
        return position._replace(direction=direction)

    @partial(jax.jit, static_argnums=(0,))
    def check_background_collision(self, state: BankHeistState, new_position: EntityPosition) -> int:
        """
        Check for collisions with the background (walls, portals).

        Returns:
            int: The maximum collision value found(255: wall, 100: portal, 200: exit, 0: empty space).
        """
        new_coords = jnp.array([new_position.x, new_position.y-1])
        new_position_bg: jnp.ndarray = jax.lax.dynamic_slice(operand=state.map_collision,
                          start_indices=new_coords, slice_sizes=COLLISION_BOX)#
        max_value = jnp.max(new_position_bg)
        return max_value

    @partial(jax.jit, static_argnums=(0,))
    def portal_handler(self, position: EntityPosition, collision: int) -> EntityPosition:
        """
        Handle portal collisions by moving the player to the corresponding portal exit.

        Returns:
            EntityPosition: The new position of the player after handling the portal collision.
        """
        side = position.x <= 80
        side = side.astype(int)
        portal_collision = collision == 100
        new_position = jax.lax.cond(portal_collision,
            lambda: position._replace(x=PORTAL_X[side]),
            lambda: position
        )
        return new_position

    @partial(jax.jit, static_argnums=(0,))
    def move(self, position: EntityPosition, direction: int, speed: int) -> EntityPosition:
        """
        Move the player in the specified direction by the specified speed.

        Returns:
            EntityPosition: The new position of the player after moving.
        """
        new_position = position
        branches = [
            lambda: new_position._replace(y=new_position.y + speed),  # DOWN
            lambda: new_position._replace(y=new_position.y - speed),  # UP
            lambda: new_position._replace(x=new_position.x + speed),  # RIGHT
            lambda: new_position._replace(x=new_position.x - speed),  # LEFT
            lambda: new_position,  # NOOP
        ]
        return jax.lax.switch(direction, branches)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        # Player step
        player_input = jnp.where(action == NOOP, state.player_position.direction, action)  # Convert NOOP to direction 4
        current_position = self.validate_input(state, state.player_position, player_input)
        new_position = self.move(current_position, current_position.direction, state.speed)
        collision = self.check_background_collision(state, new_position)
        new_position = jax.lax.cond(collision >= 255,
            lambda: current_position,
            lambda: new_position
        )
        new_position = self.portal_handler(new_position, collision)

        new_state = state._replace(
            player_position=new_position,
            )
        return state.obs_stack, new_state, 0.0, 1, {}

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BankHeistState) -> BankHeistObservation:
        return BankHeistObservation(
            player=state.player_position,
            lives=state.player_lives,
            score=state.money,
            enemies=state.enemy_positions,
            banks=state.bank_positions,
            )

def load_bankheist_sprites():
    city = aj.loadFrame(os.path.join(SPRITES_DIR, "map_1.npy"), transpose=True)
    player_side = aj.loadFrame(os.path.join(SPRITES_DIR, "player_side.npy"), transpose=True)
    player_front = aj.loadFrame(os.path.join(SPRITES_DIR, "player_front.npy"), transpose=True)
    police_side = aj.loadFrame(os.path.join(SPRITES_DIR, "police_side.npy"), transpose=True)
    police_front = aj.loadFrame(os.path.join(SPRITES_DIR, "police_front.npy"), transpose=True)
    bank = aj.loadFrame(os.path.join(SPRITES_DIR, "bank.npy"), transpose=True)

    # Add padding to front sprites so they have same dimensions as side sprites
    player_front_padded = jnp.pad(player_front, ((1,1), (0,0), (0,0)), mode='constant')
    police_front_padded = jnp.pad(police_front, ((1,1), (0,0), (0,0)), mode='constant')

    CITY_SPRITE = jnp.expand_dims(city, axis=0)
    PLAYER_SIDE_SPRITE = jnp.expand_dims(player_side, axis=0)
    PLAYER_FRONT_SPRITE = jnp.expand_dims(player_front_padded, axis=0)
    POLICE_SIDE_SPRITE = jnp.expand_dims(police_side, axis=0)
    POLICE_FRONT_SPRITE = jnp.expand_dims(police_front_padded, axis=0)
    BANK_SPRITE = jnp.expand_dims(bank, axis=0)

    return (PLAYER_SIDE_SPRITE, PLAYER_FRONT_SPRITE, POLICE_SIDE_SPRITE, POLICE_FRONT_SPRITE, BANK_SPRITE, CITY_SPRITE)

class Renderer_AtraBankisHeist:
    def __init__(self):
        (
            self.SPRITE_PLAYER_SIDE,
            self.SPRITE_PLAYER_FRONT,
            self.SPRITE_POLICE_SIDE,
            self.SPRITE_POLICE_FRONT,
            self.SPRITE_BANK,
            self.SPRITE_CITY,
        ) = load_bankheist_sprites() 

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)
        
        # Render City
        # TODO: Select Correct City to render
        frame_city = aj.get_sprite_frame(self.SPRITE_CITY, 0)
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
            state.player_position.direction == 4,
            lambda: 2,
            lambda: state.player_position.direction
        )
        player_frame = jax.lax.switch(player_direction, branches)
        raster = aj.render_at(raster, state.player_position.x, state.player_position.y, player_frame)

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