import os
from dataclasses import dataclass

from jax import config
import jax.lax
import jax.numpy as jnp
import chex
from numpy import array
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
    bullet_height: int = 5
    bullet_width: int = 5
    bullet_speed: int = 3
    cannon_height: int = 10
    cannon_width: int = 10
    cannon_y: int = 200
    cannon_x: jnp.ndarray  = field(
        default_factory=lambda: jnp.array([20, 80, 140], dtype=jnp.int32)
    )
    max_bullets: int = 20
    max_enemies: int = 20


# Each value of this class is a list.
# e.g. if i have 3 entities, then each of these lists would have a length of 3
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AtlantisState(NamedTuple):
    score: chex.Array

    # columns = [ x,  y,  dx,   type_id,  active_flag ]
    #   x, y        → position
    #   dx          → horizontal speed (positive or negative)
    #   type_id     → integer index into your enemy_specs dict
    #   active_flag → 1 if on-screen, 0 otherwise
    enemies: chex.Array # shape: (max_enemies, 5)

    # columns = [ x, y, dx, dy]. dx and dy is the velocity
    bullets: chex.Array # shape: (max_bullets, 4)
    bullets_alive: chex.Array # stores all the active bullets as bools


class AtlantisObservation(NamedTuple):
    score: jnp.ndarray
    enemy: EntityPosition
    bullet: EntityPosition

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
        blue = jnp.array([0,0,255], dtype=jnp.uint8)
        raster = jnp.broadcast_to(blue, (W,H,3))

        # get the cannons from the config
        cannon_xs = self.config.cannon_x
        cannon_y   = self.config.cannon_y

        # define cannons
        cannon_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # white blocks
        block_sprite = jnp.broadcast_to(
            cannon_color,
            (self.config.cannon_width,
             self.config.cannon_height,
             3)
        )
        # 2) define a small loop body that pastes block_sprite at each (x,y)
        def draw_one(i, img):
            x = cannon_xs[i]
            return jax.lax.dynamic_update_slice(
                img,
                block_sprite,
                (x, cannon_y, 0)     # start at (axis0=x, axis1=y, axis2=0)
            )
    
        # 3) run the loop over all cannon positions
        return jax.lax.fori_loop(
            0,
            cannon_xs.shape[0],     # 3 cannons
            draw_one,
            raster
        )

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
        # --- empty tables ---
        empty_enemies = jnp.zeros(
            (self.config.max_enemies, 5),
            dtype=jnp.int32
        )
        empty_bullets = jnp.zeros(
            (self.config.max_bullets, 4),
            dtype=jnp.int32
        )
        empty_bullets_alive = jnp.zeros(
            (self.config.max_bullets,),
            dtype=jnp.bool_
        )

        # --- initial state ---
        new_state = AtlantisState(
            score         = jnp.array(0, dtype=jnp.int32),
            enemies       = empty_enemies,
            bullets       = empty_bullets,
            bullets_alive = empty_bullets_alive
        )

        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtlantisState, action: chex.Array) -> Tuple[AtlantisState, AtlantisObservation, float, bool, AtlantisInfo]:

        def spawn_bullet(state: AtlantisState, cannon_id: int):
            # To identify which slots are free
            # bullets_alive is a boolean array. If an entry is true, then it holds an active bullet
            # ~ inverts the boolean array, such that a slot is free, when bullets_alive[i] == False
            free_slots = ~state.bullets_alive

            # get the index of the first true entry. If none is True, returns 0.
            free_idx = jnp.argmax(free_slots)

            # Compute bullet velocity based on the cannon index.
            # left = 0, center = 1, right = 2
            dx = jnp.where(
                cannon_id == 0,
                self.config.bullet_speed, # left cannon shoots
                jnp.where(
                    cannon_id == 2,
                    -self.config.bullet_speed, # right cannon shoots
                    0 # center cannon shoots
                )
            )

            #    • bullet_speed is a small integer like 3.
            #    • For c_idx==0, dx=+speed  (45° up-right)
            #      For c_idx==2, dx=–speed  (45° up-left)
            #      For c_idx==1 (centre), dx=0 and dy=–speed i.e. straight up.
            dy = -bullet_speed

            # create a 


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
