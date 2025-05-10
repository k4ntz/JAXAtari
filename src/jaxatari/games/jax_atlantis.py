import os
from dataclasses import dataclass

import jax.lax
import jax.numpy as jnp
import chex
import pygame
from typing import Dict, Any, Optional, NamedTuple, Tuple
from functools import partial

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

@dataclass(frozen=True)
class GameConfig:
    """ Game configuration parameters"""
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3

# Positions of the cannons
CANNON_X = jnp.array([20, 80, 140], dtype=jnp.int32)
CANNON_Y = 200

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AtlantisState(NamedTuple):
    score: chex.Array
    block_position: chex.Array

class AtlantisObservation(NamedTuple):
    score: jnp.ndarray
    block_position: chex.Array

class AtlantisInfo(NamedTuple):
    time: jnp.ndarray

class Renderer_AtraJaxis:
    sprites: Dict[str, Any]

    def __init__(self, config: GameConfig = None):
        self.config = config
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/atlantis"
        self.sprites = self._load_sprites()

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}

        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)


        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            # 'background_0', 'background_1', 'background_2',
            # 'ape_climb_left', 'ape_climb_right', 'ape_moving', 'ape_standing',
            # 'bell', 'ringing_bell', 'child_jump', 'child', 'coconut', 'kangaroo',
            # 'kangaroo_climb', 'kangaroo_dead', 'kangaroo_ducking',
            # 'kangaroo_jump_high', 'kangaroo_jump', 'kangaroo_lives',
            # 'kangaroo_walk', 'kangaroo_boxing',
            # 'strawberry', 'throwing_ape', 'thrown_coconut', 'time_dash',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AtlantisState):
        H = self.config.screen_height
        W = self.config.screen_width

        # Set green background
        green = jnp.array([0,255,0], dtype=jnp.uint8)
        raster = jnp.broadcast_to(green, (W,H,3))

        # define black block
        BLOCK_H, BLOCK_W = 10, 10
        block_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # white blocks
        block_sprite = jnp.broadcast_to(block_color, (BLOCK_W, BLOCK_H, 3))

        # 3) draw each block in turn
        def _draw_one(i, img):
            x, y = state.block_position[i]   # (x,y)
            return jax.lax.dynamic_update_slice(img, block_sprite, (x, y, 0))

        return jax.lax.fori_loop(0,
                                 state.block_position.shape[0],
                                 _draw_one,
                                 raster)

class JaxAtlantis(JaxEnvironment[AtlantisState, AtlantisObservation, AtlantisInfo]):
    def __init__(self, frameskip: int = 1, reward_funcs: list[callable] = None, config: GameConfig = None):
        super().__init__()
        self.config = config
        self.frameskip = frameskip
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    def reset(self) -> Tuple[AtlantisObservation, AtlantisState]:
        two_blocks = jnp.array([[50, 50],
                                [100, 150]], dtype=jnp.int32)
        new_state = AtlantisState(
            score=jnp.array(0, dtype=jnp.int32),
            block_position=two_blocks
        )
        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtlantisState, action: chex.Array) -> Tuple[AtlantisState, AtlantisObservation, float, bool, AtlantisInfo]:
        # state.block_positions has shape (N,2) where [:,0] = x, [:,1] = y
        delta = jnp.array([1, 1], dtype=jnp.int32)  # +1 in x (right), -1 in y (up)
        updated_positions = state.block_position + delta  # broadcast: adds [1, -1] to each row

        new_state = AtlantisState(
            score=jnp.array(0, dtype=jnp.int32),
            block_position=updated_positions,
        )
        observation = self._get_observation(new_state)

        reward = 0.0
        done = False

        info = AtlantisInfo(
            time=jnp.array(0, dtype=jnp.int32),
        )

        return new_state, observation, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AtlantisState) -> AtlantisObservation:
        return AtlantisObservation(
            score=jnp.array(0, dtype=jnp.int32),
            block_position=state.block_position
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AtlantisState) -> AtlantisInfo:
        """
        Placeholder info: returns zero time and empty reward array.
        """
        return AtlantisInfo(
            time=jnp.array(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        previous_state: AtlantisState,
        state: AtlantisState
    ) -> float:
        """
        Placeholder reward: always zero.
        """
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AtlantisState) -> bool:
        """
        Placeholder done: never terminates.
        """
        return False


# Keyboard inputs
def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    # up = keys[pygame.K_w] or keys[pygame.K_UP]
    # down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

    if right and fire:
        return jnp.array(Action.RIGHTFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)


def main():
    config = GameConfig()
    pygame.init()
    screen = pygame.display.set_mode((
        config.screen_width * config.scaling_factor,
        config.screen_height * config.scaling_factor
    ))
    pygame.display.set_caption("Atlantis")
    clock = pygame.time.Clock()

    game = JaxAtlantis(config=config)

    renderer = Renderer_AtraJaxis(config=config)
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    (curr_state, _) = jitted_reset()

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
                        (curr_state, _, _, _, _) = jitted_step(curr_state, action)

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                (curr_state, _, _, _, _) = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(
            screen,
            raster,
            config.scaling_factor,
            config.screen_width,
            config.screen_height
        )

        counter += 1
        # FPS
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
