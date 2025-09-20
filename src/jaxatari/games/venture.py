from typing import Optional, Dict, Any
import os

import jax.numpy as jnp
import chex
from typing import NamedTuple
import jax
from functools import partial

import pygame

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
PLAYER_SPEED = 2.0

# --- Entity Configuration (Static) ---
MAX_ARROWS = 4
MAX_MONSTERS = 8
MAX_TREASURES = 4


# --- Entity State Definitions ---

class PlayerState(NamedTuple):
    x: chex.Array  # X coordinate
    y: chex.Array  # Y coordinate
    direction: chex.Array  # 0:Up, 1:Right, 2:Down, 3:Left (for aiming)


class ArrowState(NamedTuple):
    x: chex.Array  # Shape: (MAX_ARROWS,)
    y: chex.Array
    direction: chex.Array  # Shape: (MAX_ARROWS,)
    alive: chex.Array  # Shape: (MAX_ARROWS,), bool


class MonsterState(NamedTuple):
    x: chex.Array  # Shape: (MAX_MONSTERS,)
    y: chex.Array
    type_id: chex.Array  # Shape: (MAX_MONSTERS,), to distinguish different monster types
    alive: chex.Array  # Shape: (MAX_MONSTERS,), bool


class TreasureState(NamedTuple):
    x: chex.Array  # Shape: (MAX_TREASURES,)
    y: chex.Array
    type_id: chex.Array  # Shape: (MAX_TREASURES,)
    collected: chex.Array  # Shape: (MAX_TREASURES,), bool


# --- Core Game State ---

class GameState(NamedTuple):
    """
    The main state container holding all dynamic information of the game at a specific moment.
    All members must be JAX arrays to be JIT-compatible.
    """
    player: PlayerState
    game_over: chex.Array


class JaxVenture(JaxEnvironment[GameState, chex.Array, dict]):

    def __init__(self):
        super().__init__()
        # self.config = GameConfig(...)
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.PLAYER_SPEED = PLAYER_SPEED


    def reset(self, key: jax.random.PRNGKey=None) -> tuple[chex.Array, GameState]:
        """
        Initializes the game to its starting state.
        """
        # 1. Reset the player's state
        player_state = PlayerState(
            x=jnp.array(self.SCREEN_WIDTH / 2, dtype=jnp.float32),
            y=jnp.array(self.SCREEN_HEIGHT / 2, dtype=jnp.float32),
            direction=jnp.array(0, dtype=jnp.int32)
        )

        state = GameState(player=player_state, game_over=jnp.array(False))
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> tuple[chex.Array, GameState, float, bool, dict]:
        """
        The core step function of the game. Must be a pure function and JIT-compatible.
        """

        new_player_state = self._update_player(state.player, action)
        new_state = GameState(player=new_player_state, game_over=state.game_over)
        reward = 0.0

        obs = self._get_observation(new_state)
        done = new_state.game_over
        info = {}

        return obs, new_state, reward, done, info

    # --- Helper Functions (to be implemented) ---

    def _get_observation(self, state: GameState) -> chex.Array:
        return jnp.array([state.player.x, state.player.y])

    def _update_player(self, player_state, action):
        UP_ACTIONS = jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT])
        DOWN_ACTIONS = jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT])
        LEFT_ACTIONS = jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT])
        RIGHT_ACTIONS = jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT])

        # Calculate movement delta (dx, dy)
        dy = jnp.where(jnp.isin(action, UP_ACTIONS), -self.PLAYER_SPEED, 0.0)
        dy = jnp.where(jnp.isin(action, DOWN_ACTIONS), self.PLAYER_SPEED, dy)

        dx = jnp.where(jnp.isin(action, LEFT_ACTIONS), -self.PLAYER_SPEED, 0.0)
        dx = jnp.where(jnp.isin(action, RIGHT_ACTIONS), self.PLAYER_SPEED, dx)

        # Update position
        new_x = player_state.x + dx
        new_y = player_state.y + dy

        # Clamp position to stay within screen bounds
        new_x = jnp.clip(new_x, 0, self.SCREEN_WIDTH)
        new_y = jnp.clip(new_y, 0, self.SCREEN_HEIGHT)

        # We don't update direction in this simple version
        return PlayerState(x=new_x, y=new_y, direction=player_state.direction)

    def _reset_room_entities(self, key, room_id):
        # TODO: Load monster and treasure positions from predefined level data.
        return self._create_empty_monsters(), self._create_empty_treasures()

    def _update_arrows(self, arrows_state, player_state, is_firing, cooldown):
        # TODO: Implement arrow spawning and movement physics.
        if is_firing:
            raise NotImplementedError("Arrow spawning logic is not implemented. See PROJ-46.")

        return arrows_state

    def _update_monsters(self, monsters_state, player_state, key):
        # TODO: Implement monster AI logic (e.g., pathfinding to player).
        if jnp.any(monsters_state.alive):
            raise NotImplementedError("Monster AI is not implemented. See PROJ-47.")

        return monsters_state

    def _handle_collisions(self, player, arrows, monsters, treasures, death_cooldown):
        # TODO: Implement all collision checks (Player-Monster, Arrow-Monster, etc.).
        raise NotImplementedError("Collision handling is not implemented. See PROJ-48.")

    def _create_empty_arrows(self) -> ArrowState:
        return ArrowState(
            x=jnp.zeros(MAX_ARROWS), y=jnp.zeros(MAX_ARROWS),
            direction=jnp.zeros(MAX_ARROWS, dtype=jnp.int32),
            alive=jnp.zeros(MAX_ARROWS, dtype=bool)
        )
    # ... other _create_empty... functions for monsters and treasures


def get_action_from_key() -> int:
    """Maps pygame keyboard state to a Venture action."""
    keys = pygame.key.get_pressed()

    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

    if up and right: return Action.UPRIGHT
    if up and left: return Action.UPLEFT
    if down and right: return Action.DOWNRIGHT
    if down and left: return Action.DOWNLEFT
    if up: return Action.UP
    if down: return Action.DOWN
    if left: return Action.LEFT
    if right: return Action.RIGHT

    return Action.NOOP


"""sprites please leave it at the end"""
def _load_sprites(self) -> dict[str, Any]:
    """Loads all necessary sprites from .npy files."""
sprites: Dict[str, Any] = {}       
       

import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer

class VentureRenderer(AtraJaxisRenderer):
    # Type hint for sprites dictionary
    sprites: Dict[str, Any]

    def __init__(self):
        """
        Initializes the renderer by loading sprites, including level backgrounds.

        Args:
            sprite_path: Path to the directory containing sprite .npy files.
        """
        super().__init__()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/venture"
        self.sprites = self._load_sprites()
        # Store background sprites directly for use in render function

        """
        ///////////////////////////////////////
        self.background_0 = self.sprites.get('background_0')
        self.background_1 = self.sprites.get('background_1')
        self.background_2 = self.sprites.get('background_2')
        """

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
        # background_0 is black background
        sprite_names = [
            'venture_background0', 'venture_background1',
            'venture_purple_monster', 'venture_green_monster', 'venture_dot',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite


        # pad the kangaroo and monkey sprites since they have to be used interchangeably (and jax enforces same sizes)???
        # 1. Pad BACKGROUND sprites to ensure they all have the exact same dimensions.
        # This is critical for jax.lax.switch or if they are used as the base canvas interchangeably.
        if 'venture_background0' in sprites and 'venture_background1' in sprites:
            backgrounds_to_pad = [
                sprites['venture_background0'],
                sprites['venture_background1']
            ]
            padded_backgrounds = aj.pad_to_match(backgrounds_to_pad)

            # Update the dictionary with the padded backgrounds
            sprites['venture_background0'] = padded_backgrounds[0]
            sprites['venture_background1'] = padded_backgrounds[1]

        # 2. Pad MONSTER sprites as before.
        if 'venture_purple_monster' in sprites and 'venture_green_monster' in sprites:
            monster_sprites_to_pad = [
                sprites['venture_purple_monster'],
                sprites['venture_green_monster']
            ]
            padded_monsters = aj.pad_to_match(monster_sprites_to_pad)

            sprites['venture_purple_monster'] = padded_monsters[0]
            sprites['venture_green_monster'] = padded_monsters[1]

        print("Shapes after padding and before expanding:")
        if 'venture_background0' in sprites: print(f"BG0: {sprites['venture_background0'].shape}")
        if 'venture_background1' in sprites: print(f"BG1: {sprites['venture_background1'].shape}")

        # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> chex.Array:
        """
        Renders the game state by adhering to render_at's specific API requirements.
        """
        # 1. Create a 3-CHANNEL (RGB) canvas. This is crucial for the 'raster' argument.
        canvas_rgb = jnp.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # 2. Get the background sprite, which MUST BE 4-CHANNEL (RGBA) for the 'sprite_frame' argument.
        background_rgba = aj.get_sprite_frame(self.sprites['venture_background0'], 0)

        # 3. Get the player sprite, which also MUST BE 4-CHANNEL (RGBA).
        player_rgba = aj.get_sprite_frame(self.sprites['venture_dot'], 0)

        # 4. Render the background onto the RGB canvas.
        #    raster=RGB, sprite_frame=RGBA
        canvas_rgb = aj.render_at(canvas_rgb, 0, 0, background_rgba)

        # 5. Render the player onto the resulting canvas.
        #    raster=RGB, sprite_frame=RGBA
        canvas_rgb = aj.render_at(canvas_rgb, state.player.x, state.player.y, player_rgba)

        # The final result is a 3-channel RGB image.
        return canvas_rgb


def main():
    pygame.init()

    # --- Setup ---
    env = JaxVenture()
    renderer = VentureRenderer()

    SCALING = 3
    screen = pygame.display.set_mode((env.SCREEN_WIDTH * SCALING, env.SCREEN_HEIGHT * SCALING))
    pygame.display.set_caption("JAX Venture - Movable Pink Dot")

    # JIT-compile the core functions for performance
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(renderer.render)

    # --- Initialization ---
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    clock = pygame.time.Clock()
    running = True

    # --- Game Loop ---
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 1. Get user input
        action = get_action_from_key()

        # 2. Update game state
        obs, state, reward, done, info = jitted_step(state, action)

        # 3. Render the current state to an image array
        frame = jitted_render(state)

        # 4. Display the rendered frame on the screen
        aj.update_pygame(screen, frame, SCALING, env.SCREEN_WIDTH, env.SCREEN_HEIGHT)

        # Cap the framerate
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
