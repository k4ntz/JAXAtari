from typing import NamedTuple, Tuple
import jax
import jnp
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
import chex
import os
from functools import partial
from PIL import Image
import numpy as np


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
        {'name': 'archeologist ', 'type': 'single', 'file': 'archeologist.npy'},
        {'name': 'bullet ', 'type': 'single', 'file': 'bullet.npy'},

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
        {'name': 'lives', 'type': 'single', 'pattern': 'lives.npy'},
        {'name': 'flashbangs', 'type': 'single', 'pattern': 'flashbangs.npy'},
        {'name': 'points', 'type': 'digits', 'pattern': 'lives.npy'},
        {'name': 'time', 'type': 'single', 'pattern': 'time.npy'},
    )
    return config


class Entity(NamedTuple):
    position: jnp.ndarray
    direction: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    SPEED: int = 4
    PIXEL_COLOR: Tuple[int, int, int] = (255, 255, 255)  # white

    # PLAYER_SIZE: Tuple[int, int] = (5, 10)
    # MISSILE_SIZE: Tuple[int, int] = (1, 2)

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

    creature: chex.Array

    player_lives: chex.Array
    flashbangs: chex.Array
    points: chex.Array
    time: chex.Array

    player_x: int
    player_y: int

    has_key: False
    end_reached: False

    # missile_states: chex.Array # (2, 6) array with (x, y, speed_x, speed_y, rotation, lifespan) for each missile
    # missile_rdy: chex.Array # tracks whether the player can fire a missile


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
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 3. Make one call to load and process all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)


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

        canvas[y:y + h, x:x + w] = sprite

    # ---------------------------------------------------------
    # Main render() method
    # ---------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TutankhamState):
        #pseudo for no errors
        player_mask = 0
        camera_offset = 0
        not_vanishing = True
        floor_checks = [1]
        ghost_frame = 0
        indices_to_update = 0
        new_color_ids = 0



        # 1. Start with the static blue background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self.jr.render_at(
            raster, state.player[0], state.player[1] - camera_offset,
            player_mask, flip_offset=self.FLIP_OFFSETS['player_group']
        )
        raster = jax.lax.cond(
            floor_checks[0] & not_vanishing,
            lambda r: self.jr.render_at_clipped(
                r, state.ghost[0], state.ghost[1] - camera_offset,
                self.SHAPE_MASKS['ghost_group'][ghost_frame],
                flip_offset=self.FLIP_OFFSETS['ghost_group']
            ),
            lambda r: r,
            raster
        )
        # 2. Render Player
        player_frame = jnp.where(state.stun_duration > 0, state.stun_duration % 8 + 1, state.player_direction[1])
        player_mask = self.SHAPE_MASKS['player_group'][player_frame]
        raster = self.jr.render_at(
            raster, state.player[0], state.player[1] - camera_offset,
            player_mask, flip_offset=self.FLIP_OFFSETS['player_group']
        )
        # 2.5 Animations
        # 3. Render Walls
        # 4. Render Teleporter and Spawner
        # 5. Render Treasures
        # 6. Render Bullets
        # 7. Render Enemies
        # 8. Render UI
        # 9. Final Palette Lookup
        return self.jr.render_from_palette(
            raster,
            self.PALETTE,
            indices_to_update=indices_to_update,
            new_color_ids=new_color_ids
        )
    def srender(self, state):

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

    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: TutankhamState) -> TutankhamState:
        return state._replace()

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2

        state = TutankhamState(player_x=start_x, player_y=start_y, player_lives=3)
        return state, state

    def lose_life(self, state: TutankhamState) -> TutankhamState:
        """
        Handle collision with creatures by reducing player lives and resetting player position.

        Args:
            state: Current game state

        Returns:
            TutankhamState: Updated state with reduced lives and reset player position
        """
        # Reduce player lives by 1
        new_player_lives = state.player_lives - 1

        # Reset player to last position
        player_x = 1
        player_y = 1
        default_player_position = jnp.array([12, 78]).astype(jnp.int32)
        new_player = state.player._replace(position=default_player_position)

        return state._replace(
            player_lives=new_player_lives,
            player=new_player,

        )

    # -----------------------------
    # Step logic (pure Python)
    # -----------------------------
    def unlock_door(self, state: TutankhamState):
        return state.has_key

    def _get_done(self, state: TutankhamState):
        return state.player_lives == 0 or state.end_reached

    def player_step(self):
        pass

    def item_step(self):
        pass

    def enemy_step(self):
        pass

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

        # Step 1: Player Mechanics

        (player, player_direction, stun_duration, match_duration, matches_used,
         item_dropped, stairs_active, fire_button_active, lives, game_ends) = self.player_step(state, action)

        # Step 2: Item Mechanics
        (item_held, scepter, urn_left, urn_middle, urn_right,
         urn_left_middle, urn_middle_right, urn_left_right, urn) = self.item_step(state, item_dropped)

        # Step 3: Enemy Mechanics
        ghost, spider, bat, current_nodes, previous_nodes, chasing = self.enemy_step(state)

        # Step 4: Increase Step Counter
        step_counter_reset_condition = False
        step_counter = jax.lax.cond(
            step_counter_reset_condition,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        new_state = TutankhamState(
            #TODO fill new state with states
        )

        done = self._get_done(new_state)
        # env_reward = self._get_reward(state, new_state)
        # info = self._get_info(new_state)
        # observation = self._get_observation(new_state)

        reward = 0.0
        info = None

        # return observation, new_state, env_reward, done, info
        return state, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def check_wall_collision(self, pos, size):
        """Check collision between an entity and the wall"""

        # Because the wall sprite is not at (0,0)
        pos = jnp.array([pos[0], pos[1] - self.consts.WALL_Y_OFFSET])

        collision_top_left = self.consts.WALL[pos[1]][pos[0]]
        collision_top_right = self.consts.WALL[pos[1]][pos[0] + size[0] - 1]
        collision_bottom_left = self.consts.WALL[pos[1] + size[1] - 1][pos[0]]
        collision_bottom_right = self.consts.WALL[pos[1] + size[1] - 1][pos[0] + size[0] - 1]

        return jnp.any(
            jnp.array([collision_top_left, collision_top_right, collision_bottom_right, collision_bottom_left]))
        # return False

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
