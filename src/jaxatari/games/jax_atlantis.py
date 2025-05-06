import os
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from typing import Dict, Any, Optional, NamedTuple, Tuple
from functools import partial

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

# pygame window size
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Constants for
WIDTH = 160
HEIGHT = 210

# Action constants
NOOP = 0
FIRE = 1
RIGHTFIRE = 2
LEFTFIRE = 3


######
# Hier die Klassendefinitionen
######
class AtlantisState(NamedTuple):
    score: chex.Array
    obs_stack: chex.ArrayTree

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class AtlantisObservation(NamedTuple):
    player: EntityPosition

class AtlantisInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class Renderer_AtraJaxis:
    sprites: Dict[str, Any]

    def __init__(self):
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
        # Blank rgb frame
        return jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

class JaxAtlantis(JaxEnvironment[AtlantisState, AtlantisObservation, AtlantisInfo]):
    def __init__(self, frameskip: int = 1, reward_funcs: list[callable] = None):
        # Andere rufen hier super().__init__() auf. warum? Die functionis leer
        self.frameskip = frameskip
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            NOOP,
            FIRE,
            RIGHTFIRE,
            LEFTFIRE,
        }
        # Keine Ahnung was das macht. Probieren wir mal 10...
        self.obs_size = 10

    def reset(self) -> AtlantisState:
        state = AtlantisState(
            score=0,
            obs_stack=None
        )
        initial_obs = self._get_observation(state)


        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        new_state = state._replace(obs_stack=initial_obs)
        return new_state, initial_obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtlantisState, action: chex.Array) -> Tuple[AtlantisState, AtlantisObservation, float, bool, AtlantisInfo]:
        """ PLACEHOLDER !!"""
        # Create a dummy player observation at (0,0) with zero size
        player = EntityPosition(
            x=jnp.array(0, dtype=jnp.int32),
            y=jnp.array(0, dtype=jnp.int32),
            width=jnp.array(0, dtype=jnp.int32),
            height=jnp.array(0, dtype=jnp.int32)
        )
        observation = AtlantisObservation(player=player)

        # Dummy reward and termination flag
        reward = 0.0
        done = False

        # Dummy info with zero time and zero rewards
        info = AtlantisInfo(
            time=jnp.array(0, dtype=jnp.int32),
            all_rewards=jnp.zeros((1,), dtype=jnp.float32)
        )

        return state, observation, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AtlantisState) -> AtlantisObservation:
        """
        Placeholder observation: wraps state into minimal AtlantisObservation.
        """
        player = EntityPosition(
            x=jnp.array(0, dtype=jnp.int32),
            y=jnp.array(0, dtype=jnp.int32),
            width=jnp.array(0, dtype=jnp.int32),
            height=jnp.array(0, dtype=jnp.int32)
        )
        return AtlantisObservation(player=player)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AtlantisState) -> AtlantisInfo:
        """
        Placeholder info: returns zero time and empty reward array.
        """
        return AtlantisInfo(
            time=jnp.array(0, dtype=jnp.int32),
            all_rewards=jnp.zeros((1,), dtype=jnp.float32)
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
        return jnp.array(RIGHTFIRE)
    if left and fire:
        return jnp.array(LEFTFIRE)
    if fire:
        return jnp.array(FIRE)

    return jnp.array(NOOP)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Atlantis")
    clock = pygame.time.Clock()
    scaling_factor = 3

    game = JaxAtlantis()

    renderer = Renderer_AtraJaxis()
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

        aj.update_pygame(screen, raster, scaling_factor, WINDOW_WIDTH, WINDOW_HEIGHT)

        counter += 1
        pygame.time.Clock().tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
