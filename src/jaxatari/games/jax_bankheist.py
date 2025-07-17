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

LEFT = 3
RIGHT = 2
UP = 1
DOWN = 0

EMPTY_SPACE_ID = 0
WALLS_ID = 1



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

class BankHeistObservation(NamedTuple):
    player: EntityPosition
    mothership: EntityPosition
    enemy_1: EntityPosition
    enemy_2: EntityPosition
    enemy_3: EntityPosition
    enemy_4: EntityPosition
    enemy_5: EntityPosition
    enemy_6: EntityPosition
    enemy_projectile: EntityPosition
    lives: jnp.ndarray
    score: jnp.ndarray


class BankHeistInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxBankHeist(JaxEnvironment[BankHeistState, BankHeistObservation, BankHeistInfo]):
    
    def __init__(self):
        super().__init__()
    
    def reset(self) -> BankHeistState:
        # Minimal state initialization
        state = BankHeistState(
            fuel = jnp.array(100).astype(jnp.int32),
            lives = jnp.array(4).astype(jnp.int32)
        )
        obs = self._get_observation(state)
        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
        obs_stack = jax.tree.map(expand_and_copy, obs)
        state = state._replace(obs_stack=obs_stack)
        return  obs_stack, state
    
    partial(jax.jit, static_argnums=(0,))
    def step(self, state: BankHeistState, action: chex.Array) -> Tuple[BankHeistState, BankHeistObservation, float, bool, BankHeistInfo]:
        # Player step
      pass

def load_bankheist_sprites():
  pass

class Renderer_AtraBankisHeist:
    def __init__(self):
        (
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            self.SPRITE_BANK
        ) = load_bankheist_sprites() 

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
      raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)
      return

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
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

