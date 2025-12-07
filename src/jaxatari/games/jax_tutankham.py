from typing import NamedTuple, Tuple
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
import chex
import os


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for BankHeist.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # Define file lists for groups
    tombs = [f"tomb_{i + 1}.npy" for i in range(4)]

    # Define the sprites
    config = (
        # Backgrounds (loaded as a group)
        # Note: The 'background' type is not used here, as the city map is the primary background.
        # We will treat 'tombs' as our base background sprites.
        {'name': 'tombs', 'type': 'group', 'files': tombs},

        # Player (loaded as single sprites for manual padding)
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},

        # Creatures (loaded as single sprites for manual padding)
        {'name': 'snake', 'type': 'single', 'file': 'snake.npy'},
        {'name': 'scorpion', 'type': 'single', 'file': 'scorpion.npy'},
        {'name': 'bat', 'type': 'single', 'file': 'bat.npy'},
        {'name': 'turtle', 'type': 'single', 'file': 'turtle.npy'},
        {'name': 'jackel', 'type': 'single', 'file': 'jackel.npy'},
        {'name': 'condor', 'type': 'single', 'file': 'condor.npy'},
        {'name': 'lion', 'type': 'single', 'file': 'lion.npy'},
        {'name': 'moth', 'type': 'single', 'file': 'moth.npy'},
        {'name': 'virus', 'type': 'single', 'file': 'virus.npy'},
        {'name': 'monkey', 'type': 'single', 'file': 'monkey.npy'},
        {'name': 'mystery', 'type': 'single', 'file': 'mystery.npy'},
        {'name': 'weapon', 'type': 'single', 'file': 'weapon.npy'},

        # Treasures
        {'name': 'key', 'type': 'single', 'file': 'key.npy'},
        {'name': 'crown', 'type': 'single', 'file': 'crown.npy'},
        {'name': 'ring', 'type': 'single', 'file': 'ring.npy'},
        {'name': 'ruby', 'type': 'single', 'file': 'ruby.npy'},
        {'name': 'chalice', 'type': 'single', 'file': 'chalice.npy'},
        {'name': 'emerald', 'type': 'single', 'file': 'emerald.npy'},
        {'name': 'goblet', 'type': 'single', 'file': 'goblet.npy'},
        {'name': 'bust', 'type': 'single', 'file': 'bust.npy'},
        {'name': 'trident', 'type': 'single', 'file': 'trident.npy'},
        {'name': 'herb', 'type': 'single', 'file': 'herb.npy'},
        {'name': 'diamond', 'type': 'single', 'file': 'diamond.npy'},
        {'name': 'candelabra', 'type': 'single', 'file': 'candelabra.npy'},
        {'name': 'amulet', 'type': 'single', 'file': 'amulet.npy'},
        {'name': 'fan', 'type': 'single', 'file': 'fan.npy'},
        {'name': 'crystal', 'type': 'single', 'file': 'crystal.npy'},
        {'name': 'zircon', 'type': 'single', 'file': 'zircon.npy'},
        {'name': 'dagger', 'type': 'single', 'file': 'dagger.npy'},

        # UI
        {'name': 'lives', 'type': 'digits', 'pattern': 'lives.npy'},
        {'name': 'flashbangs', 'type': 'digits', 'pattern': 'flashbangs.npy'},
        {'name': 'points', 'type': 'digits', 'pattern': 'lives.npy'},

    )

    return config

class Entity(NamedTuple):
    position: jnp.ndarray
    direction: jnp.ndarray


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    SPEED: int = 4
    PIXEL_COLOR: Tuple[int, int, int] = (255, 255, 255)  # white

    #PLAYER_SIZE: Tuple[int, int] = (5, 10)
    #MISSILE_SIZE: Tuple[int, int] = (1, 2)

    # Asset config baked into constants
    ASSET_CONFIG: tuple = _get_default_asset_config()

# ---------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------
class TutankhamState(NamedTuple):
    level: chex.Array
    player: Entity
    creature_positions: Entity
    treasure_positions: Entity

    player_lives: chex.Array
    flashbangs: chex.Array
    points: chex.Array
    time: chex.Array

    player_x: int
    player_y: int

    #missile_states: chex.Array # (2, 6) array with (x, y, speed_x, speed_y, rotation, lifespan) for each missile
    #missile_rdy: chex.Array # tracks whether the player can fire a missile


# ---------------------------------------------------------------------
# Renderer (No JAX)
# ---------------------------------------------------------------------

class sTutankhamRenderer(JAXGameRenderer):
    def render(self, state: TutankhamState) -> np.ndarray:
        frame = np.zeros(
            (self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=np.uint8
        )

        x = min(max(state.player_x, 0), self.consts.WIDTH - 1)
        y = min(max(state.player_y, 0), self.consts.HEIGHT - 1)

        frame[y, x] = np.array(self.consts.PIXEL_COLOR, dtype=np.uint8)
        return frame

from PIL import Image
import numpy as np

class TutankhamRenderer(JAXGameRenderer):
    def __init__(self, consts: TutankhamConstants = None):
        super().__init__()
        self.consts = consts or TutankhamConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham"

        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )

        self.width = TutankhamConstants.WIDTH
        self.height = TutankhamConstants.HEIGHT
        self.tile_size = tile_size
        self.sprites = {}   # name → numpy array (RGB)

    # ---------------------------------------------------------
    # Sprite Loader
    # ---------------------------------------------------------
    def load_sprite(self, name, path, size=None):
        if size is None:
            size = (self.tile_size, self.tile_size)
        img = Image.open(path).convert("RGB").resize(size)
        self.sprites[name] = np.array(img, dtype=np.uint8)

    # ---------------------------------------------------------
    # Draw a sprite into canvas
    # ---------------------------------------------------------
    def draw_sprite(self, canvas, sprite_name, x, y):
        if sprite_name not in self.sprites:
            raise ValueError(f"Sprite '{sprite_name}' not loaded!")

        sprite = self.sprites[sprite_name]
        h, w, _ = sprite.shape

        # Clipping
        x = max(0, min(self.width - w, x))
        y = max(0, min(self.height - h, y))

        canvas[y:y+h, x:x+w] = sprite

    # ---------------------------------------------------------
    # Main render() method
    # ---------------------------------------------------------
    def render(self, state):
        """
        state muss folgende Keys enthalten:
        - tilemap: 2D array von Tile-IDs
        - player: {"x":..., "y":...}
        - enemies: [{"x":..., "y":...}, ...]
        - projectiles: [{"x":..., "y":...}, ...]
        - items: [{"x":..., "y":..., "type":...}]
        """

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # -------------------------------------------------
        # 1. TILEMAP ZEICHNEN (Labyrinth)
        # -------------------------------------------------
        tilemap = state["tilemap"]
        for row_idx, row in enumerate(tilemap):
            for col_idx, tile_id in enumerate(row):
                tile_name = f"tile_{tile_id}"
                if tile_name in self.sprites:  # floor, wall, door, etc.
                    x = col_idx * self.tile_size
                    y = row_idx * self.tile_size
                    self.draw_sprite(canvas, tile_name, x, y)

        # -------------------------------------------------
        # 2. SPIELER ZEICHNEN
        # -------------------------------------------------
        px = state["player"]["x"]
        py = state["player"]["y"]
        self.draw_sprite(canvas, "player", px, py)

        # -------------------------------------------------
        # 3. GEGNER ZEICHNEN
        # -------------------------------------------------
        for enemy in state.get("enemies", []):
            self.draw_sprite(canvas, "enemy", enemy["x"], enemy["y"])

        # -------------------------------------------------
        # 4. PROJEKTILE ZEICHNEN
        # -------------------------------------------------
        for proj in state.get("projectiles", []):
            self.draw_sprite(canvas, "projectile", proj["x"], proj["y"])

        # -------------------------------------------------
        # 5. ITEMS ZEICHNEN (Schlüssel, Schatz, etc.)
        # -------------------------------------------------
        for item in state.get("items", []):
            sprite_name = f"item_{item['type']}"
            self.draw_sprite(canvas, sprite_name, item["x"], item["y"])

        return Image.fromarray(canvas)


# ---------------------------------------------------------------------
# Environment (No JAX)
# ---------------------------------------------------------------------
class JaxTutankham(JaxEnvironment):
    def __init__(self):
        consts = TutankhamConstants()
        super().__init__(consts)
        self.renderer = TutankhamRenderer()
        self.consts = consts

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ]

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2

        state = TutankhamState(player_x=start_x, player_y=start_y)
        return state, state

    # -----------------------------
    # Step logic (pure Python)
    # -----------------------------
    def step(self, state: TutankhamState, action: int):

        x, y = state.player_x, state.player_y

        if action == Action.LEFT:
            x -= self.consts.SPEED
        elif action == Action.RIGHT:
            x += self.consts.SPEED
        elif action == Action.UP:
            y -= self.consts.SPEED
        elif action == Action.DOWN:
            y += self.consts.SPEED

        # Clip bounds
        x = max(0, min(x, self.consts.WIDTH - 1))
        y = max(0, min(y, self.consts.HEIGHT - 1))

        state = TutankhamState(player_x=x, player_y=y)

        reward = 0.0
        done = False
        info = None

        return state, state, reward, done, info

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self, state: TutankhamState) -> np.ndarray:
        return self.renderer.render(state)

    # -----------------------------
    # Action & Observation Space
    # -----------------------------
    def action_space(self):
        return Discrete(len(self.action_set))

    def observation_space(self):
        return Box(
            low=0,
            high=max(self.consts.WIDTH, self.consts.HEIGHT),
            shape=(2,),
            dtype=np.int32,
        )
