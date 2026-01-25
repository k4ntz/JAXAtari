"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct

import jaxatari.rendering.jax_rendering_utils_legacy as jru
import jaxatari.rendering.jax_rendering_utils as render_utils
import time
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Procedural black background
    procedural_bg = jnp.zeros((210, 160, 4), dtype=jnp.uint8).at[:, :, 3].set(255)

    # Procedural colors for palette swapping (constant colors from CentipedeConstants) [redefinition due to current limitation of sprite loading]
    all_colors = [
        [181, 83, 40],    # ORANGE 
        [184, 70, 162],   # PINK
        [146, 70, 192],   # PURPLE
        [45, 50, 184],    # DARK_BLUE
        [187, 187, 53],   # YELLOW
        [184, 50, 50],    # RED
        [110, 156, 66],   # GREEN
        [84, 138, 210],   # LIGHT_BLUE
        [66, 72, 200],    # DARK_PURPLE
        [162, 162, 42],   # DARK_YELLOW
    ]
    procedural_colors_data = jnp.array(
        [list(c) + [255] for c in all_colors], dtype=jnp.uint8
    ).reshape(-1, 1, 1, 4)
    
    return {
        'background': procedural_bg,
        'recolor_palette': procedural_colors_data,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Centipede.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    
    # Poisoned mushroom files
    pmush_files = [f'poisoned_mushrooms/{i}.npy' for i in range(1, 17)]

    return (
        # Procedural assets
        {'name': 'background', 'type': 'background', 'data': static_procedural['background']},
        {'name': 'recolor_palette', 'type': 'procedural', 'data': static_procedural['recolor_palette']},
        # Single sprites
        {'name': 'player', 'type': 'single', 'file': 'player/player.npy'},
        {'name': 'player_spell', 'type': 'single', 'file': 'player_spell/player_spell.npy'},
        {'name': 'mushroom', 'type': 'single', 'file': 'mushrooms/mushroom.npy'},
        {'name': 'centipede', 'type': 'single', 'file': 'centipede/segment.npy'},
        {'name': 'spider_300', 'type': 'single', 'file': 'spider_scores/300.npy'},
        {'name': 'spider_600', 'type': 'single', 'file': 'spider_scores/600.npy'},
        {'name': 'spider_900', 'type': 'single', 'file': 'spider_scores/900.npy'},
        {'name': 'bottom_border', 'type': 'single', 'file': 'ui/bottom_border.npy'},
        {'name': 'life_indicator', 'type': 'single', 'file': 'ui/wand.npy'},
        # Groups (for auto-padding)
        {'name': 'spider_group', 'type': 'group', 'files': ['spider/1.npy', 'spider/2.npy', 'spider/3.npy', 'spider/4.npy']},
        {'name': 'flea_group', 'type': 'group', 'files': ['flea/1.npy', 'flea/2.npy']},
        {'name': 'scorpion_group', 'type': 'group', 'files': ['scorpion/1.npy', 'scorpion/2.npy']},
        {'name': 'sparks', 'type': 'group', 'files': ['sparks/1.npy', 'sparks/2.npy', 'sparks/3.npy', 'sparks/4.npy']},
        {'name': 'poisoned_mushroom_group', 'type': 'group', 'files': pmush_files},
        # Digits
        {'name': 'digits', 'type': 'digits', 'pattern': 'big_numbers/{}.npy'},
    )

class CentipedeConstants(struct.PyTreeNode):
    # -------- Game constants --------
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    SCALING_FACTOR: int = struct.field(pytree_node=False, default=6)

    ## -------- Player constants --------
    PLAYER_START_X: int = struct.field(pytree_node=False, default=76)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=172)
    PLAYER_BOUNDS: Tuple[Tuple[int, int], Tuple[int, int]] = struct.field(pytree_node=False, default_factory=lambda: ((16, 140), (141, 172)))

    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 9))

    PLAYER_Y_VALUES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([141, 145, 147, 150, 154, 156, 159, 163, 165, 168, 172]))      # Double to not need extra state value

    ## -------- Player spell constants --------
    PLAYER_SPELL_SPEED: int = struct.field(pytree_node=False, default=9)

    PLAYER_SPELL_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (1, 8))

    ## -------- Starting Pattern (X -> placed, O -> not placed, P -> placed and poisoned) --------
    MUSHROOM_STARTING_PATTERN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: [
        "OOOOOOOOOOOOOOOO",
        "OOOOOOOOXOOOOXOO",
        "OOOOOOOOOXOOOOXO",
        "OOOOOOOOOOXXOOOO",
        "OOOXOOOOOOOOOOOO",
        "OXOOOOOOOOOOOOOO",
        "OOOOOOOXOOOOOOOO",
        "OOOOOOXOOOOOOOXO",
        "OOOXXOOOOXOOOOOO",
        "OOOOOOOXOOOOOOOO",
        "OOOOOOOOOOOOXOOO",
        "OOOOXOOOOOOOOOOO",
        "OOOOOOOOOOOOOXOO",
        "OOOOOOOOOOOXOOOX",
        "OOOOOOOOOOOOOOXO",
        "OOOOXOOXOOOOOOOO",
        "OXOOOXOOOOOOOOOO",
        "OOOOXOOOOOOOOOOX",
        "OOOOOOOOOOOOOOOO",
    ])

    ## -------- Mushroom constants --------
    MAX_MUSHROOMS: int = struct.field(pytree_node=False, default=304)             # Default 304 (19*16) | Maximum number of mushrooms that can appear at the same time
    MUSHROOM_NUMBER_OF_ROWS: int = struct.field(pytree_node=False, default=19)    # Default 19 | Number of rows -> Determines value of MAX_MUSHROOMS
    MUSHROOM_NUMBER_OF_COLS: int = struct.field(pytree_node=False, default=16)    # Default 16 | Number of mushrooms per row -> Determines value of MAX_MUSHROOMS
    MUSHROOM_X_SPACING: int = struct.field(pytree_node=False, default=8)      #
    MUSHROOM_Y_SPACING: int = struct.field(pytree_node=False, default=9)
    MUSHROOM_COLUMN_START_EVEN: int = struct.field(pytree_node=False, default=20)
    MUSHROOM_COLUMN_START_ODD: int = struct.field(pytree_node=False, default=16)
    MUSHROOM_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 3))
    MUSHROOM_HITBOX_Y_OFFSET: int = struct.field(pytree_node=False, default=6)

    ## -------- Centipede constants --------
    MAX_SEGMENTS: int = struct.field(pytree_node=False, default=9)
    SEGMENT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 6))

    ## -------- Spider constants --------
    SPIDER_X_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 133]))
    SPIDER_Y_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([115, 124, 133, 142, 151, 160, 169, 178]))
    SPIDER_MOVE_PROBABILITY: float = struct.field(pytree_node=False, default=0.2)
    SPIDER_MIN_SPAWN_FRAMES: int = struct.field(pytree_node=False, default=55)
    SPIDER_MAX_SPAWN_FRAMES: int = struct.field(pytree_node=False, default=355)
    SPIDER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 6))
    SPIDER_CLOSE_RANGE: int = struct.field(pytree_node=False, default=16)
    SPIDER_MID_RANGE: int = struct.field(pytree_node=False, default=32)

    ## -------- Scorpion constants --------
    SCORPION_X_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 133]))
    SCORPION_Y_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([7, 16, 25, 34, 43, 52, 61, 70, 79, 88, 97, 106, 115, 124, 133]))
    SCORPION_MIN_SPAWN_FRAMES: int = struct.field(pytree_node=False, default=355)
    SCORPION_MAX_SPAWN_FRAMES: int = struct.field(pytree_node=False, default=2000)
    SCORPION_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 6))
    SCORPION_POINTS: int = struct.field(pytree_node=False, default=1000)

    ## -------- Flea constants --------
    FLEA_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 6))
    FLEA_SPAWN_MUSHROOM_PROBABILITY: float = struct.field(pytree_node=False, default=0.5)
    FLEA_POINTS: int = struct.field(pytree_node=False, default=200)

    ## -------- Death animation constants --------
    DEATH_ANIMATION_MUSHROOM_THRESHOLD: int = struct.field(pytree_node=False, default=64)        # 4 Frames * 4 Sprites * 4 Repetitions

    ## -------- Color constants --------
    ORANGE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(181, 83, 40))#B55328    # Mushrooms lvl1
    DARK_ORANGE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(198, 108, 58))#C66C3A
    PINK: Tuple[int, int, int] = struct.field(pytree_node=False, default=(184, 70, 162))#B846A2      # Centipede lvl1
    GREEN: Tuple[int, int, int] = struct.field(pytree_node=False, default=(110, 156, 66))#6E9C42     # Border lvl1
    LIGHT_PURPLE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(188, 144, 252))#BC90FC  # UI Elements
    PURPLE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(146, 70, 192))#9246C0        # Spider
    DARK_PURPLE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(66, 72, 200))#4248C8
    LIGHT_BLUE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(84, 138, 210))#548AD2    # Border lvl2
    DARK_BLUE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(45, 50, 184))#2D32B8     # Mushrooms lvl2
    LIGHT_RED: Tuple[int, int, int] = struct.field(pytree_node=False, default=(200, 72, 72))#C84848
    RED: Tuple[int, int, int] = struct.field(pytree_node=False, default=(184, 50, 50))#B83232       # Centipede lvl2
    YELLOW: Tuple[int, int, int] = struct.field(pytree_node=False, default=(187, 187, 53))#BBBB35
    DARK_YELLOW: Tuple[int, int, int] = struct.field(pytree_node=False, default=(162, 162, 42))#A2A22A

    ## -------- Sprite Frames --------
    SPRITE_PLAYER_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_PLAYER_SPELL_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_CENTIPEDE_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_MUSHROOM_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_SPIDER_FRAMES: int = struct.field(pytree_node=False, default=4)
    SPRITE_SPIDER_300_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_SPIDER_600_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_SPIDER_900_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_FLEA_FRAMES: int = struct.field(pytree_node=False, default=2)
    SPRITE_SCORPION_FRAMES: int = struct.field(pytree_node=False, default=2)
    SPRITE_SPARKS_FRAMES: int = struct.field(pytree_node=False, default=4)
    SPRITE_BOTTOM_BORDER_FRAMES: int = struct.field(pytree_node=False, default=1)
    SPRITE_POISONED_MUSHROOMS_FRAMES: int = struct.field(pytree_node=False, default=16)

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

    # -------- Centipede States --------

@struct.dataclass
class CentipedeState:
    player_x: chex.Array
    player_y: chex.Array
    player_velocity_x: chex.Array
    player_spell: chex.Array  # (1, 3): x, y, is_alive
    mushroom_positions: chex.Array # (304, 4): x, y, is_poisoned, lives; 304 mushrooms in total
    centipede_position: chex.Array # (9, 5): x, y, speed(horizontal), movement(vertical), status/is_head; 9 segments in total
    centipede_spawn_timer: chex.Array # Frames until new centipede head spawns
    spider_position: chex.Array # (1, 3): x, y, direction
    spider_spawn_timer: chex.Array # Frames until new spider spawns
    spider_points: chex.Array # (1, 2): sprite, timeout
    flea_position: chex.Array # (1, 3) array for flea: (x, y, lives), 2 lives, speed doubles after 1 hit
    flea_spawn_timer: chex.Array # Frames until new flea spawns
    scorpion_position: chex.Array # (1, 3) array for scorpion: (x, y, direction)
    scorpion_spawn_timer: chex.Array
    score: chex.Array
    lives: chex.Array
    wave: chex.Array # (1, 2): logical wave, ui wave
    step_counter: chex.Array
    death_counter: chex.Array
    spark_position: chex.Array
    rng_key: chex.PRNGKey

@struct.dataclass
class PlayerEntity:
    x: jnp.ndarray
    y: jnp.ndarray
    o: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

@struct.dataclass
class EntityPosition:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    
    active: jnp.ndarray

@struct.dataclass
class CentipedeObservation:
    player: PlayerEntity
    mushrooms: jnp.ndarray      # Shape (MAX_MUSHROOMS, 5) - MAX_MUSHROOMS mushrooms, each with x,y,w,h,active
    centipede: jnp.ndarray      # Shape (MAX_SEGMENTS, 5) - MAX_SEGMENTS Centipede segments, each with x,y,w,h,active
    spider: EntityPosition      # Shape (5,) - one spider with x,y,w,h,active
    flea: EntityPosition        # Shape (5,) - one flea with x,y,w,h,active
    scorpion: EntityPosition    # Shape (5,) - one scorpion with x,y,w,h,active
    player_spell: EntityPosition    # Shape (5,) - one spell with x,y,w,h,active
    score: jnp.ndarray
    lives: jnp.ndarray

@struct.dataclass
class CentipedeInfo:
    wave: jnp.ndarray
    step_counter: jnp.ndarray

# -------- Game Logic --------

class JaxCentipede(JaxEnvironment[CentipedeState, CentipedeObservation, CentipedeInfo, CentipedeConstants]):
    # Minimal ALE action set (from scripts/action_space_helper.py)
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

    def __init__(self, consts: CentipedeConstants = None):
        consts = consts or CentipedeConstants()
        super().__init__(consts)
        self.obs_size = 6 + 304 * 5 + 9 * 5 + 5 + 5 + 5 + 5 + 1 + 1
        self.renderer = CentipedeRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CentipedeState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.array([entity.x], dtype=jnp.int32),
            jnp.array([entity.y], dtype=jnp.int32),
            jnp.array([entity.width], dtype=jnp.int32),
            jnp.array([entity.height], dtype=jnp.int32),
            jnp.array([entity.active], dtype=jnp.int32)
        ])

    def flatten_player_entity(self, entity: PlayerEntity) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.array([entity.x], dtype=jnp.int32),
            jnp.array([entity.y], dtype=jnp.int32),
            jnp.array([entity.o], dtype=jnp.int32),
            jnp.array([entity.width], dtype=jnp.int32),
            jnp.array([entity.height], dtype=jnp.int32),
            jnp.array([entity.active], dtype=jnp.int32)
        ])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CentipedeObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_entity(obs.player),
            obs.mushrooms.flatten().astype(jnp.int32),
            obs.centipede.flatten().astype(jnp.int32),
            self.flatten_entity_position(obs.spider),
            self.flatten_entity_position(obs.flea),
            self.flatten_entity_position(obs.scorpion),
            self.flatten_entity_position(obs.player_spell),
            obs.score.flatten().astype(jnp.int32),
            obs.lives.flatten().astype(jnp.int32),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Centipede.
        The observation contains:
        - player: PlayerEntity (x, y, o, width, height, active)
        - mushrooms: array of shape (304, 5) with x,y,width,height,active for each mushroom
        - centipede: array of shape (9, 5) with x,y,width,height,active for each segment
        - spider: EntityPosition (x, y, width, height, active)
        - flea: EntityPosition (x, y, width, height, active)
        - scorpion: EntityPosition (x, y, width, height, active)
        - player_spell: EntityPosition (x, y, width, height, active)
        - score: int (0-999999)
        - lives: int (0-3)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "o": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "mushrooms": spaces.Box(low=0, high=210, shape=(304, 5), dtype=jnp.int32),
            "centipede": spaces.Box(low=0, high=210, shape=(9, 5), dtype=jnp.int32),
            "spider": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "flea": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "scorpion": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_spell": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Centipede.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: CentipedeState) -> CentipedeObservation:
        # Create player (already scalar, no need for vectorization)
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            o=jnp.array(0),  # No orientation in Centipede, set to 0
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
            active=jnp.array(1),  # Player is always active
        )

        # Define a function to convert mushroom positions to entity format
        def convert_to_mushroom_entity(pos):
            x, y, is_poisoned, lives = pos
            return jnp.array([
                x,  # x position
                y,  # y position
                self.consts.MUSHROOM_SIZE[0],  # width
                self.consts.MUSHROOM_SIZE[1],  # height
                lives > 0,  # active flag
            ])

        # Mushrooms
        mushrooms = jax.vmap(convert_to_mushroom_entity)(
            state.mushroom_positions
        )

        # Define a function to convert centipede segments to entity format
        def convert_to_centipede_entity(pos):
            x, y, speed_h, movement_v, status = pos
            return jnp.array([
                x.astype(jnp.int32),  # x position
                y.astype(jnp.int32),  # y position
                self.consts.SEGMENT_SIZE[0],  # width
                self.consts.SEGMENT_SIZE[1],  # height
                status != 0,  # active flag assuming status indicates activity
            ])

        # Centipede segments
        centipede = jax.vmap(convert_to_centipede_entity)(
            state.centipede_position
        )

        # Spider (scalar)
        spider_pos = state.spider_position
        spider = EntityPosition(
            x=spider_pos[0],
            y=spider_pos[1],
            width=jnp.array(self.consts.SPIDER_SIZE[0]),
            height=jnp.array(self.consts.SPIDER_SIZE[1]),
            active=spider_pos[0] != 0,
        )

        # Flea (scalar)
        flea_pos = state.flea_position
        flea = EntityPosition(
            x=flea_pos[0].astype(jnp.int32),
            y=flea_pos[1].astype(jnp.int32),
            width=jnp.array(self.consts.FLEA_SIZE[0]),
            height=jnp.array(self.consts.FLEA_SIZE[1]),
            active=flea_pos[2] > 0,
        )

        # Scorpion (scalar)
        scorpion_pos = state.scorpion_position
        scorpion = EntityPosition(
            x=scorpion_pos[0],
            y=scorpion_pos[1],
            width=jnp.array(self.consts.SCORPION_SIZE[0]),
            height=jnp.array(self.consts.SCORPION_SIZE[1]),
            active=scorpion_pos[0] != 0,
        )

        # Player spell (scalar)
        spell_pos = state.player_spell
        player_spell = EntityPosition(
            x=spell_pos[0],
            y=spell_pos[1],
            width=jnp.array(self.consts.PLAYER_SPELL_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SPELL_SIZE[1]),
            active=spell_pos[2] != 0,
        )

        # Return observation
        return CentipedeObservation(
            player=player,
            mushrooms=mushrooms,
            centipede=centipede,
            spider=spider,
            flea=flea,
            scorpion=scorpion,
            player_spell=player_spell,
            score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CentipedeState) -> CentipedeInfo:
        return CentipedeInfo(
            wave=state.wave[1],
            step_counter=state.step_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: CentipedeState, state: CentipedeState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: CentipedeState) -> bool:
        return state.lives < 0

    # -------- Helper Functions --------

    @partial(jax.jit, static_argnums=(0, ))
    def check_collision_single(self, pos1, size1, pos2, size2):
        """Check collision between two single entities"""
        # Calculate edges for rectangle 1
        rect1_left = pos1[0]
        rect1_right = pos1[0] + size1[0]
        rect1_top = pos1[1]
        rect1_bottom = pos1[1] + size1[1]

        # Calculate edges for rectangle 2
        rect2_left = pos2[0]
        rect2_right = pos2[0] + size2[0]
        rect2_top = pos2[1]
        rect2_bottom = pos2[1] + size2[1]

        # Check overlap
        horizontal_overlap = jnp.logical_and(
            rect1_left < rect2_right,
            rect1_right > rect2_left
        )

        vertical_overlap = jnp.logical_and(
            rect1_top < rect2_bottom,
            rect1_bottom > rect2_top
        )

        return jnp.logical_and(horizontal_overlap, vertical_overlap)

    @partial(jax.jit, static_argnums=(0, ))
    def get_mushroom_index(self, pos: chex.Array) -> chex.Array:
        row_idx = (pos[1] - 7) / 9
        odd_row = pos[1] % 2 == 0
        col_idx = jnp.where(odd_row, pos[0], pos[0] - 4) / 8
        return row_idx * 16 + col_idx - 2

    # -------- Logic Functions --------

    ## -------- Centipede Move Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def centipede_step(
            self,
            centipede_state: chex.Array,
            mushrooms_positions: chex.Array,
    ) -> chex.Array:

        # --- Utility: detect collision with mushroom ---
        def check_mushroom_collision(segment, mushroom, seg_speed):
            direction = jnp.where(seg_speed > 0, 6, -6)
            collision = jnp.logical_and(
                self.check_collision_single(
                    pos1=jnp.array([segment[0], segment[1]]),
                    size1=self.consts.SEGMENT_SIZE,
                    pos2=mushroom[:2],
                    size2=self.consts.MUSHROOM_SIZE,
                ),
                self.check_collision_single(
                    pos1=jnp.array([segment[0] + direction, segment[1]]),
                    size1=self.consts.SEGMENT_SIZE,
                    pos2=mushroom[:2],
                    size2=self.consts.MUSHROOM_SIZE,
                ),
            )
            return jnp.where(mushroom[3] > 0, collision, False)  # mushroom alive?

        # --- Poisoned zig-zag behaviour ---
        def poisoned_step(segment):
            speed = segment[2] * 0.5
            new_x = segment[0]
            new_y = segment[1]

            # check walls & mushrooms
            hit_left = jnp.logical_and(new_x <= self.consts.PLAYER_BOUNDS[0][0], segment[2] < 0)
            hit_right = jnp.logical_and(new_x >= self.consts.PLAYER_BOUNDS[0][1], segment[2] > 0)
            mushroom_collision = jnp.any(
                jax.vmap(lambda m: check_mushroom_collision(segment, m, segment[2]))(mushrooms_positions))

            move_down = jnp.logical_or(
                jnp.logical_or(hit_left, hit_right),
                jnp.logical_or(mushroom_collision, (segment[0] + 1) % 4 == 0)
            )

            def descend():
                new_y = jnp.where(
                    segment[1] < 176,
                    segment[1] + self.consts.MUSHROOM_Y_SPACING,
                    segment[1] - self.consts.MUSHROOM_Y_SPACING
                )
                new_vertical = jnp.where(segment[1] < 176, 2, -1)
                return new_x - speed, new_y, -segment[2], new_vertical

            def keep_horizontal():
                return new_x + speed, new_y, segment[2], segment[3].astype(jnp.int32)

            # if diving down
            def poisoned_down():
                new_x2, new_y2, new_horiz, new_vertical = jax.lax.cond(move_down, descend, keep_horizontal)
                return new_x2, new_y2, new_horiz, new_vertical

            new_x, new_y, new_horiz, new_vertical = poisoned_down()

            new_status = jnp.where(segment[4] == 1.5, 2, segment[4])
            return jnp.array([new_x, new_y, new_horiz, new_vertical, new_status]), jnp.array(0)

        # --- Normal centipede step (your old move_segment without poisoned branch) ---
        def normal_step(segment, turn_around):
            moving_left = segment[2] < 0

            def step_horizontal():
                speed = segment[2] * 0.5
                new_x = segment[0] + speed
                return jnp.array([new_x, segment[1], segment[2], segment[3], segment[4]]), jnp.array(0)

            def step_vertical():
                moving_down = jnp.greater(segment[3], 0)
                y_dif = jnp.where(moving_down, self.consts.MUSHROOM_Y_SPACING, -self.consts.MUSHROOM_Y_SPACING)
                new_y = segment[1] + y_dif

                new_vertical = jnp.where(
                    jnp.logical_or(new_y >= 176, jnp.logical_and(segment[3] < 0, new_y <= 131)),
                    -segment[3],
                    segment[3],
                )
                new_status = jnp.where(segment[4] == 1.5, 2, segment[4])

                return jnp.array([segment[0], new_y, -segment[2], new_vertical, new_status]), jnp.array(0)

            def step_turn_around():
                new_speed = -segment[2]
                speed = new_speed * 0.5
                new_x = segment[0] + speed
                return jnp.array([new_x, segment[1], new_speed, segment[3], segment[4]]), jnp.array(1)

            # detect collisions
            collision = jnp.any(
                jax.vmap(lambda m: check_mushroom_collision(segment, m, segment[2]))(mushrooms_positions))

            move_down = jnp.logical_or(
                collision,
                jnp.logical_or(
                    jnp.logical_and(segment[0] <= self.consts.PLAYER_BOUNDS[0][0], moving_left),
                    jnp.logical_and(segment[0] >= self.consts.PLAYER_BOUNDS[0][1], jnp.invert(moving_left)),
                ),
            )

            return jax.lax.cond(
                move_down,
                lambda: jax.lax.cond(jnp.logical_and(segment[1] >= 176, turn_around == 1), step_turn_around,
                                     step_vertical),
                step_horizontal,
            )

        # --- Dispatcher: poisoned overrides normal ---
        def move_segment(segment, turn_around):
            # check poisoned collision now
            poisoned_collision = jnp.any(
                jax.vmap(lambda m: jnp.logical_and(check_mushroom_collision(segment, m, segment[2]), m[2] == 1))(
                    mushrooms_positions)
            )
            is_already_poisoned = jnp.logical_or(segment[3] == 2, segment[3] == -2)
            poisoned_active = jnp.logical_or(poisoned_collision, is_already_poisoned)

            return jax.lax.cond(
                poisoned_active,
                lambda: poisoned_step(segment),
                lambda: normal_step(segment, turn_around),
            )

        def should_turn_around(i, carry):
            groups, same_group = carry
            this_seg = centipede_state[i]
            next_seg = centipede_state[i + 1]

            is_head = jnp.where(this_seg[4] == 2, True, False)

            def head_case():
                def same_group_head():
                    return groups.at[i].set(1), True

                def dif_group():
                    return groups, False

                dist = jnp.linalg.norm(this_seg[:2] - next_seg[:2])
                return jax.lax.cond(jnp.less_equal(dist, 10), same_group_head, dif_group)

            def same_group_tail():
                return groups, True

            return jax.lax.cond(is_head, head_case, same_group_tail)

        # --- Run update over all segments ---
        init_carry = jnp.zeros_like(centipede_state[:, 0], dtype=jnp.int32), False
        turn_around, _ = jax.lax.fori_loop(0, centipede_state.shape[0] - 1, should_turn_around, init_carry)     # afaik not vmappable

        new_state, segment_split = jax.vmap(move_segment)(centipede_state, turn_around)
        segment_split = jnp.roll(segment_split, 1)

        def set_new_status(seg, split):
            return jnp.where(split == 1, seg.at[4].set(1.5), seg)

        return jax.vmap(set_new_status)(new_state, segment_split)

    @partial(jax.jit, static_argnums=(0,))
    def handle_centipede_segment_spawn(self, centipede_timer, centipede_position) -> tuple[chex.Array, chex.Array]:
        timer_threshold = 192 - (centipede_timer // 1000) * 8
        spawn = centipede_timer % 1000 >= timer_threshold
        new_timer = jnp.where(
            jnp.sum(jnp.clip(centipede_position[:, 4], 0, 1)) < 9,
            jnp.where(
                centipede_timer == 0,
                jnp.where(
                    jnp.max(centipede_position[:, 1]) >= 176,
                    jnp.array(1),
                    jnp.array(0),
                ),
                jnp.where(
                    spawn,
                    (centipede_timer // 1000 + 1) * 1000,
                    jnp.array(centipede_timer + 1),
                )
            ),
            jnp.array(0),
        )

        def spawn_new_segment():
            min_idx = jnp.argmin(centipede_position[:, 4])      # when called, this should be guaranteed to point to a zero element
            direction = jnp.sum(jnp.clip(centipede_position[:, 2], -1, 1)) < 0     # true = dir.left, false = dir.right
            return centipede_position.at[min_idx].set(
                jnp.where(
                    direction,
                    jnp.array([140, 131, -2, 1, 2]),
                    jnp.array([16, 131, 2, 1, 2]),
                )
            )

        return new_timer, jax.lax.cond(spawn, spawn_new_segment, lambda: centipede_position)

    ## -------- Spider Move Logic -------- ##
    def spider_step(
            self,
            spider_x: chex.Array,
            spider_y: chex.Array,
            spider_direction: chex.Array,
            step_counter: chex.Array,
            key: chex.PRNGKey
    ) -> chex.Array:
        """Moves Spider one Step further with random x-movement and periodic y-movement"""

        # Split key in two parts, one for x and one for y
        key_x, key_y = jax.random.split(key)

        # X movement of spider
        move_x = jax.random.bernoulli(key_x, self.consts.SPIDER_MOVE_PROBABILITY)  # True = bewegen
        new_x = jnp.where(move_x, spider_x + spider_direction, spider_x)

        # Check if left or right border is reached
        stop_left = (spider_direction == -1) & (new_x < 16)
        stop_right = (spider_direction == +1) & (new_x > 133)

        new_direction = jnp.where(stop_left | stop_right, 0, spider_direction)

        # Y movement of spider
        move_y = (step_counter % 8 == 7)

        def update_y(spider_y):
            idx = jnp.argwhere(self.consts.SPIDER_Y_POSITIONS == spider_y, size=1).squeeze()

            can_go_up = idx > 0
            can_go_down = idx < len(self.consts.SPIDER_Y_POSITIONS) - 1

            rand = jax.random.bernoulli(key_y)

            dy = jnp.where(~can_go_down, -1,
                           jnp.where(~can_go_up, +1,
                                     jnp.where(rand, -1, +1)))

            new_idx = idx + dy
            return self.consts.SPIDER_Y_POSITIONS[new_idx]

        new_y = jax.lax.cond(
            move_y,
            update_y,
            lambda y: y,
            spider_y
        )

        return jnp.stack([new_x, new_y, new_direction])

    @partial(jax.jit, static_argnums=(0,))
    def spider_alive_step(
            self,
            spider_x: chex.Array,
            spider_y: chex.Array,
            spider_dir: chex.Array,
            step_counter: chex.Array,
            key_step: chex.PRNGKey,
            spawn_timer: chex.Array,
    ) -> tuple[chex.Array, int]:
        new_spider = self.spider_step(spider_x, spider_y, spider_dir, step_counter, key_step)
        return new_spider, spawn_timer

    @partial(jax.jit, static_argnums=(0,))
    def spider_dead_step(
            self,
            spider_position: chex.Array,
            spawn_timer: int,
            key_spawn: chex.PRNGKey,
    ) -> tuple[chex.Array, int]:
        new_timer = spawn_timer - 1

        def respawn():
            new_spider = self.initialize_spider_position(key_spawn)
            next_timer = jax.random.randint(
                key_spawn,
                (),
                self.consts.SPIDER_MIN_SPAWN_FRAMES,
                self.consts.SPIDER_MAX_SPAWN_FRAMES + 1
            )
            return new_spider, next_timer

        def wait():
            return jnp.array([spider_position[0], spider_position[1], 0]), new_timer

        return jax.lax.cond(new_timer <= 0, respawn, wait)

    ## -------- Flea Move Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def flea_step(
            self,
            flea_position: chex.Array,
            flea_spawn_timer: chex.Array,
            mushroom_positions: chex.Array,
            wave: chex.Array,
            rng_key: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        activate = jnp.count_nonzero(jnp.where(mushroom_positions[:, 1] > 97, mushroom_positions[:, 3], 0)) < 5

        def handle_position():

            def spawn_new():
                init_x = (
                        self.consts.MUSHROOM_COLUMN_START_ODD +
                        jax.random.randint(rng_key, (), 0, 15) * self.consts.MUSHROOM_X_SPACING
                )
                init_y = 5.0
                return jnp.array([init_x, init_y, 2.0])

            def move_flea():
                old_y = flea_position[1]
                v = jnp.where(flea_position[2] == 2, 0.125, 0.25)
                big_step = jnp.logical_or(
                    jnp.equal(jnp.floor(old_y) + 0.5, old_y),
                    jnp.equal(jnp.floor(old_y) + 0.625, old_y)
                )
                new_y = jnp.where(big_step, old_y + 8.5, old_y + v)
                return jnp.where(new_y < 185, flea_position.at[1].set(new_y), jnp.zeros_like(flea_position))

            return jax.lax.cond(flea_position[2] == 0, lambda: spawn_new(), lambda: move_flea())

        new_spawn_timer = jnp.where(
            jnp.logical_and(activate, jnp.logical_and(flea_position[2] == 0, wave[0] != 0)),
            jnp.mod(flea_spawn_timer, 30) + 1,
            0
        )
        # jax.debug.print("{x}, {y}", x=new_spawn_timer, y=flea_position[1])

        new_position = jax.lax.cond(
            jnp.logical_or(flea_position[2] != 0, flea_spawn_timer == 30),
            lambda: handle_position(),
            lambda: flea_position
        )

        def spawn_mushroom():
            pos = jnp.array([new_position[0], jnp.floor(new_position[1]) + 2])
            mush_idx = self.get_mushroom_index(pos)
            can_spawn = jnp.logical_and(
                jnp.floor(new_position[1]) == new_position[1],
                jnp.logical_and(jnp.floor(mush_idx) == mush_idx, mush_idx < mushroom_positions.shape[0])
            )
            should_spawn = jax.random.randint(rng_key, (), 0, 10) < 10 * self.consts.FLEA_SPAWN_MUSHROOM_PROBABILITY

            def spawn(idx):
                mush_idx = jnp.array(idx, dtype=jnp.int32)
                return mushroom_positions.at[mush_idx, 3].set(3)

            return jax.lax.cond(jnp.logical_and(should_spawn, can_spawn), lambda: spawn(mush_idx), lambda: mushroom_positions)

        return new_position, new_spawn_timer, spawn_mushroom()

    ## -------- Scorpion Move Logic -------- ##
    def scorpion_step(
            self,
            scorpion_x: chex.Array,
            scorpion_y: chex.Array,
            scorpion_direction: chex.Array,
            scorpion_speed: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:

        new_x = scorpion_x + scorpion_direction * scorpion_speed

        stop_left = (scorpion_direction == -1) & (new_x < self.consts.SCORPION_X_POSITIONS[0])
        stop_right = (scorpion_direction == +1) & (new_x > self.consts.SCORPION_X_POSITIONS[1])

        vanished = (stop_left | stop_right).astype(jnp.int32)
        new_direction = jnp.where(vanished == 1, 0, scorpion_direction)

        new_scorpion = jnp.stack([new_x, scorpion_y, new_direction, scorpion_speed])

        return new_scorpion, vanished

    def scorpion_alive_step(
            self,
            scorpion_x: chex.Array,
            scorpion_y: chex.Array,
            scorpion_dir: chex.Array,
            scorpion_speed: chex.Array,
            key_step: chex.PRNGKey,
            spawn_timer: chex.Array,
            mushroom_positions: chex.Array,
            poison_stop_flag: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:

        new_scorpion, vanished = self.scorpion_step(scorpion_x, scorpion_y, scorpion_dir, scorpion_speed)

        def on_vanish(args):
            new_scorpion, old_spawn_timer, mushrooms = args

            key_local = key_step
            next_timer = jax.random.randint(
                key_local,
                (),
                self.consts.SCORPION_MIN_SPAWN_FRAMES,
                self.consts.SCORPION_MAX_SPAWN_FRAMES + 1
            )

            updated_mush = jax.lax.cond(
                poison_stop_flag == 0,  # direction != 0
                lambda: self.poison_mushrooms(mushrooms, new_scorpion[1]),
                lambda: mushrooms
            )

            dead_scorpion = jnp.array([new_scorpion[0], new_scorpion[1], 0, new_scorpion[3]])

            return dead_scorpion, next_timer, updated_mush

        def no_vanish(args):
            new_scorpion, old_spawn_timer, mushrooms = args
            return new_scorpion, old_spawn_timer, mushrooms

        result = jax.lax.cond(
            vanished == 1,
            on_vanish,
            no_vanish,
            operand=(new_scorpion, spawn_timer, mushroom_positions)
        )

        return result

    def scorpion_dead_step(
            self,
            scorpion_position: chex.Array,
            spawn_timer: chex.Array,
            key_spawn: chex.PRNGKey,
            wave: chex.Array,
            mushroom_positions: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:

        new_timer = spawn_timer - 1

        def respawn():
            def do_spawn():
                key_pos, key_rand = jax.random.split(key_spawn)
                new_scorpion = self.initialize_scorpion_position(key_pos, score)
                next_timer = jax.random.randint(
                    key_rand,
                    (),
                    self.consts.SCORPION_MIN_SPAWN_FRAMES,
                    self.consts.SCORPION_MAX_SPAWN_FRAMES + 1
                )
                return new_scorpion, next_timer, mushroom_positions

            def skip_spawn():
                next_timer = jax.random.randint(
                    key_spawn,
                    (),
                    self.consts.SCORPION_MIN_SPAWN_FRAMES,
                    self.consts.SCORPION_MAX_SPAWN_FRAMES + 1
                )
                return jnp.array(
                    [
                        scorpion_position[0],
                        scorpion_position[1],
                        0,
                        scorpion_position[3]
                    ]
                ), next_timer, mushroom_positions

            return jax.lax.cond(
                wave[1] >= 3,
                do_spawn,
                skip_spawn
            )

        def wait():
            return jnp.array(
                [
                    scorpion_position[0],
                    scorpion_position[1],
                    0,
                    scorpion_position[3]
                ]
            ), new_timer, mushroom_positions

        #jax.debug.print("Scorpion spawn_timer={t} dir={d}", t=spawn_timer, d=scorpion_position[2])

        return jax.lax.cond(new_timer <= 0, respawn, wait)

    ## -------- Scorpion Poison Logic -------- ##
    def poison_mushrooms(
            self,
            mushroom_positions: chex.Array,
            scorpion_y: float
    ) -> chex.Array:

        same_row = mushroom_positions[:, 1] == scorpion_y
        alive = mushroom_positions[:, 3] > 0
        to_poison = same_row & alive


        poisoned_col = jnp.where(to_poison, 1, mushroom_positions[:, 2])


        updated = jnp.concatenate([
            mushroom_positions[:, :2],
            poisoned_col.reshape(-1, 1),
            mushroom_positions[:, 3:]
        ], axis=1)

        return updated

    ## -------- Spell Mushroom Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def check_spell_mushroom_collision(
        self,
        spell_state: chex.Array,
        mushroom_positions: chex.Array,
        score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        def check_single_mushroom(is_alive, mushroom, score):
            def no_hit():
                return is_alive, mushroom, score

            def check_hit():
                mush_pos = mushroom[:2]
                mush_hp = mushroom[3]

                collision = self.check_collision_single(
                    pos1=jnp.array([spell_pos_x, spell_pos_y + self.consts.MUSHROOM_HITBOX_Y_OFFSET]),
                    size1=self.consts.PLAYER_SPELL_SIZE,
                    pos2=mush_pos,
                    size2=self.consts.MUSHROOM_SIZE
                )

                def on_hit():
                    new_hp = mush_hp - 1
                    updated_mushroom_position = mushroom.at[3].set(new_hp)
                    new_score = jnp.where(new_hp == 0, score + 1, score)
                    return False, updated_mushroom_position, new_score

                def check_hp():
                    return jax.lax.cond(mush_hp > 0, on_hit, lambda: (is_alive, mushroom, score))

                return jax.lax.cond(collision, check_hp, lambda: (is_alive, mushroom, score))

            return jax.lax.cond(is_alive != 0, check_hit, no_hit)

        spell_active, updated_mushrooms, updated_score = jax.vmap(
            lambda m: check_single_mushroom(spell_is_alive, m, score)
        )(mushroom_positions)

        spell_active = jnp.invert(jnp.any(jnp.invert(spell_active)))
        return spell_state.at[2].set(jnp.where(spell_active, 1, 0)), updated_mushrooms, jnp.max(updated_score)

    ## -------- Spider Mushroom Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spider_mushroom_collision(
            self,
            spider_position: chex.Array,
            mushroom_positions: chex.Array,  # shape (304, 4): x, y, is_poisoned, lives
    ) -> chex.Array:
        """
        Detects collisions between spider and mushrooms.
        Returns updated mushroom positions (lives set to 0 if hit by spider).
        """

        spider_x, spider_y, spider_dir = spider_position
        spider_alive = spider_dir != 0

        def no_collision():
            return mushroom_positions

        def check_hit():
            # Prüfe pro Mushroom Kollision mit der Spinne
            def collide_single(mushroom):
                x, y, is_poisoned, lives = mushroom

                collision = self.check_collision_single(
                    pos1=jnp.array([spider_x + 4, spider_y - 2]),
                    size1=(2, 4), # mushrooms do not always react to the spider bumping into them so the spider frame to check has to be smaller than SPIDER_SIZE (so we take MUSHROOM_SIZE so that two mushrooms cannot be hit at the same time)
                    pos2=jnp.array([x, y]),
                    size2=self.consts.MUSHROOM_SIZE,
                )

                # Bei Kollision direkt lives auf 0 setzen
                new_lives = jnp.where(collision, 0, lives)

                return jnp.array([x, y, is_poisoned, new_lives])

            new_mushrooms = jax.vmap(collide_single)(mushroom_positions)
            return new_mushrooms

        return jax.lax.cond(spider_alive, check_hit, no_collision)

    ## -------- Centipede Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_centipede_collision(
            self,
            spell_state: chex.Array,
            centipede_position: chex.Array,
            mushroom_positions: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        spell_active = spell_state[2] != 0

        def no_hit():
            return (
                jnp.repeat(spell_active, centipede_position.shape[0]),
                centipede_position,
                jnp.repeat(0, centipede_position.shape[0]),
                jnp.repeat(0, centipede_position.shape[0]),
                jnp.repeat(-1, centipede_position.shape[0])
            )

        def check_single_segment(is_alive, seg):
            seg_pos = seg[:2]

            collision = self.check_collision_single(
                pos1=jnp.array([spell_state[0], spell_state[1]]),
                size1=self.consts.PLAYER_SPELL_SIZE,
                pos2=seg_pos,
                size2=self.consts.SEGMENT_SIZE,
            )

            def on_hit():
                mush_y = seg[1] + 2
                odd_mush_row = seg[1] % 2 == 0
                mush_x = jnp.where(
                    odd_mush_row,
                    jnp.where(
                        seg[2] > 0,
                        jnp.ceil(seg[0] / 8) * 8,
                        jnp.floor(seg[0] / 8) * 8,
                    ),
                    jnp.where(
                        seg[2] > 0,
                        jnp.ceil(seg[0] / 8) * 8 + 4,
                        jnp.floor(seg[0] / 8) * 8 + 4,
                    )
                )
                out_of_border = jnp.where(
                    odd_mush_row,
                    jnp.logical_or(
                        jnp.logical_and(seg[2] > 0, mush_x > 136),
                        jnp.logical_and(seg[2] < 0, mush_x < 16)
                    ),
                    jnp.logical_or(
                        jnp.logical_and(seg[2] > 0, mush_x > 140),
                        jnp.logical_and(seg[2] < 0, mush_x < 20)
                    )
                )
                idx = jnp.where(out_of_border, -1, self.get_mushroom_index(jnp.array([mush_x, mush_y])))
                return (
                    False,
                    jnp.zeros_like(seg),
                    jnp.where(seg[4] == 2, 100, 10),
                    jnp.array(1),
                    jnp.array(idx, dtype=jnp.int32)
                )

            return jax.lax.cond(collision, on_hit, lambda: (is_alive, seg, 0, jnp.array(0), jnp.array(-1)))

        check = jax.vmap(lambda s: check_single_segment(spell_active, s), in_axes=0)

        (
            spell_active,
            new_centipede_position,
            new_score,
            segment_hit,
            mush_idx
        ) = jax.lax.cond(spell_active != 0, lambda: check(centipede_position), no_hit)
        spell_active = jnp.invert(jnp.any(jnp.invert(spell_active)))

        new_score = jnp.sum(new_score)
        new_heads = jnp.roll(segment_hit, 1)
        mush_idx = jnp.max(mush_idx)
        new_mushroom_positions = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    mush_idx >= 0,
                    mush_idx < self.consts.MAX_MUSHROOMS
                ),
                mushroom_positions[mush_idx, 3] == 0
            ),
            mushroom_positions.at[mush_idx, 3].set(3),
            mushroom_positions
        )

        def set_new_status(seg, new):     # change value of hit following segment to head
            return jnp.where(jnp.logical_and(new == 1, seg[4] != 0), seg.at[4].set(2), seg)

        return (
            spell_state.at[2].set(jnp.where(spell_active, 1, 0)),
            jax.vmap(set_new_status)(new_centipede_position, new_heads),
            new_mushroom_positions,
            score + new_score
        )

    ## -------- Spider Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_spider_collision(
            self,
            spell_state: chex.Array,
            spider_position: chex.Array,
            score: chex.Array,
            player_y: chex.Array,
            spider_points: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Detects collision between spell and spider.
        Points are determined by the distance from player to spider on hit
        - close range: 900 Points
        - middle range: 600 Points
        - far range: 300 Points
        Additionally returns a spider_timer_sprite as int32[2] array.
        """

        # Check if spell is still active
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        # Check if spider is still active
        spider_x, spider_y, spider_dir = spider_position
        spider_alive = spider_dir != 0

        # Default return (no collision, no sprite)
        def no_collision():
            return spell_state, spider_position, score, spider_points

        def check_hit():
            collision = self.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([spider_x + 2, spider_y - 2]),
                size2=self.consts.SPIDER_SIZE,
            )

            def on_hit():
                # Distance from player to spider
                dist = jnp.abs(player_y - spider_y)

                # Points determined by distance
                points = jnp.where(
                    dist < self.consts.SPIDER_CLOSE_RANGE,
                    900,
                    jnp.where(
                        dist < self.consts.SPIDER_MID_RANGE,
                        600,
                        300,
                    ),
                )

                # Sprite determined by distance
                spider_timer_sprite = jnp.select(
                    [
                        dist < self.consts.SPIDER_CLOSE_RANGE,
                        dist < self.consts.SPIDER_MID_RANGE,
                    ],
                    [
                        jnp.array([3, 55], dtype=jnp.int32),
                        jnp.array([2, 55], dtype=jnp.int32),
                    ],
                    default=jnp.array([1, 55], dtype=jnp.int32),
                )

                new_spell = spell_state.at[2].set(0)
                new_spider = jnp.array([spider_x, spider_y, 0])
                new_score = score + points
                return new_spell, new_spider, new_score, spider_timer_sprite

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, spider_alive),
            check_hit,
            no_collision,
        )

    ## -------- Flea Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_flea_collision(
            self,
            spell_state: chex.Array,
            flea_position: chex.Array,
            flea_spawn_counter: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:

        # Spell info
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        flea_x, flea_y, flea_lives = flea_position
        flea_alive = flea_lives != 0

        # Default: no collision
        def no_collision():
            return spell_state, flea_position, flea_spawn_counter, score

        def check_hit():
            # Collision check
            collision = self.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([flea_x, flea_y]),
                size2=self.consts.FLEA_SIZE,
            )

            def on_hit():
                new_spell = spell_state.at[2].set(0)
                new_flea_lives = flea_lives - 1
                new_flea = jnp.where(new_flea_lives == 0, jnp.array([0.0, 0.0, 0.0]), flea_position.at[2].set(new_flea_lives))
                new_score = jnp.where(new_flea_lives == 0, score + self.consts.FLEA_POINTS, score)
                return new_spell, new_flea, jnp.array(29), new_score

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, flea_alive),
            check_hit,
            no_collision
        )


    ## -------- Scorpion Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_scorpion_collision(
            self,
            spell_state: chex.Array,
            scorpion_position: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Detects collision between spell and scorpion.
        No sprite is returned.
        If hit:
          - scorpion dir is set to 0
          - poisoning is stopped for this frame (poison_stop_flag=1)
          - score is increased by SCORPION_POINTS
        Returns:
          spell_state, new_scorpion_position, new_score, poison_stop_flag
        """

        # Spell info
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        # Scorpion info
        scorpion_x, scorpion_y, scorpion_dir, scorpion_speed = scorpion_position
        scorpion_alive = scorpion_dir != 0

        # Default: no collision
        def no_collision():
            return spell_state, scorpion_position, score, jnp.array(0, dtype=jnp.int32)

        def check_hit():
            # Collision check
            collision = self.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([scorpion_x, scorpion_y]),
                size2=self.consts.SCORPION_SIZE,
            )

            def on_hit():
                new_spell = spell_state.at[2].set(0)
                new_scorpion = jnp.array([scorpion_x, scorpion_y, 0, scorpion_speed])
                new_score = score + self.consts.SCORPION_POINTS
                return new_spell, new_scorpion, new_score, jnp.array(1, dtype=jnp.int32)  # poison_stop_flag

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, scorpion_alive),
            check_hit,
            no_collision
        )

    ## -------- Player Enemy Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_player_enemy_collision(
            self,
            player_x,
            player_y,
            centipede_position,
            spider_position,
            flea_position,
    ) -> chex.Array:

        # Get centipede params
        centipede_is_alive = jnp.any(centipede_position[:, 3] != 0)

        # Get spider params
        spider_x = spider_position[0]
        spider_y = spider_position[1]
        spider_is_alive = jnp.where(spider_position[2] != 0, True, False)

        # Get flea params
        flea_x = flea_position[0]
        flea_y = flea_position[1]
        flea_is_alive = jnp.where(flea_position[2] > 0, True, False)

        # Default: no collision
        def no_collision():
            return jnp.array(0)

        def check_hit():

            # Check Centipede Player collision
            def single_collision(c_xy, active):
                return jnp.where(
                    active != 0,
                    self.check_collision_single(
                        pos1=jnp.array([player_x, player_y + 1]),
                        size1=(4, 8),
                        pos2=c_xy,
                        size2=self.consts.SEGMENT_SIZE,
                    ),
                    False
                )

            centipede_collision = jax.vmap(single_collision)(
                centipede_position[:, :2],
                centipede_position[:, 3]
            )

            centipede_collision_any = jnp.any(centipede_collision)

            # Check Spider Player collision
            spider_collision = jnp.where(
                spider_is_alive,
                self.check_collision_single(
                    pos1=jnp.array([player_x, player_y]),
                    size1=self.consts.PLAYER_SIZE,
                    pos2=jnp.array([spider_x + 2, spider_y - 2]),
                    size2=self.consts.SPIDER_SIZE,
                ),
                False
            )

            # Check Flea Player collision
            flea_collision = jnp.where(
                flea_is_alive,
                self.check_collision_single(
                    pos1=jnp.array([player_x, player_y]),
                    size1=self.consts.PLAYER_SIZE,
                    pos2=jnp.array([flea_x, flea_y]),
                    size2=self.consts.FLEA_SIZE,
                ),
                False
            )




            collision = jnp.logical_or(centipede_collision_any, jnp.logical_or(flea_collision, spider_collision))

            def on_hit():
                return jnp.array(-1)

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_or(centipede_is_alive, jnp.logical_or(spider_is_alive, flea_is_alive)),
            check_hit,
            no_collision
        )

    ## -------- Mushroom Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def initialize_mushroom_positions(self) -> chex.Array:
        rows = jnp.arange(self.consts.MUSHROOM_NUMBER_OF_ROWS)
        cols = jnp.arange(self.consts.MUSHROOM_NUMBER_OF_COLS)

        # --- Build char_code grid from Python pattern ---
        # Convert each row into padded ord codes
        def row_to_codes(row: str) -> jnp.ndarray:
            # Pad row with "O" if shorter than max cols
            padded = row.ljust(self.consts.MUSHROOM_NUMBER_OF_COLS, "O")
            return jnp.array([ord(c) for c in padded], dtype=jnp.int32)

        # Run row_to_codes across all rows in the Python list
        char_codes = jnp.stack([row_to_codes(r) for r in self.consts.MUSHROOM_STARTING_PATTERN])

        # --- Lives and poison pattern from char codes ---
        def char_to_lives(ch: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(
                (ch == ord("X")) | (ch == ord("x")) | (ch == ord("P")) | (ch == ord("p")),
                3, 0
            )

        def char_to_poison(ch: jnp.ndarray) -> jnp.ndarray:
            return jnp.where((ch == ord("P")) | (ch == ord("p")), 1, 0)

        lives_pattern   = jax.vmap(jax.vmap(char_to_lives))(char_codes)
        poison_pattern  = jax.vmap(jax.vmap(char_to_poison))(char_codes)

        # Pad patterns to required number of rows
        pad_rows = max(0, self.consts.MUSHROOM_NUMBER_OF_ROWS - lives_pattern.shape[0])
        lives_pattern  = jnp.pad(lives_pattern,  ((0, pad_rows), (0, 0)), constant_values=0)
        poison_pattern = jnp.pad(poison_pattern, ((0, pad_rows), (0, 0)), constant_values=0)

        # --- Per-cell computation ---
        def cell_fn(row, col):
            row_is_even = (row % 2) == 0
            column_start = jnp.where(
                row_is_even,
                self.consts.MUSHROOM_COLUMN_START_EVEN,
                self.consts.MUSHROOM_COLUMN_START_ODD,
            )
            x = column_start + self.consts.MUSHROOM_X_SPACING * col
            y = row * self.consts.MUSHROOM_Y_SPACING + 7
            lives   = lives_pattern[row, col]
            poisoned = poison_pattern[row, col]
            return jnp.array([x, y, poisoned, lives], dtype=jnp.int32)

        # Vectorize across grid with nested vmaps
        grid = jax.vmap(lambda r: jax.vmap(lambda c: cell_fn(r, c))(cols))(rows)

        # Flatten to (N*M, 4)
        return grid.reshape(-1, 4)

    ## -------- Spider Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def initialize_spider_position(self, key: chex.PRNGKey) -> chex.Array:
        idx_x = jax.random.randint(key, (), 0, 2)
        x = self.consts.SPIDER_X_POSITIONS[idx_x]

        direction = jnp.where(x == self.consts.SPIDER_X_POSITIONS[1], -1, +1)

        idx_y = jax.random.randint(key, (), 0, len(self.consts.SPIDER_Y_POSITIONS))
        y = self.consts.SPIDER_Y_POSITIONS[idx_y]

        return jnp.stack([x, y, direction])

    ## -------- Scorpion Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def initialize_scorpion_position(self, key: chex.PRNGKey, score:chex.Array) -> chex.Array:
        idx_x = jax.random.randint(key, (), 0, 2)
        x = self.consts.SCORPION_X_POSITIONS[idx_x]

        direction = jnp.where(x == self.consts.SCORPION_X_POSITIONS[1], -1, +1)

        idx_y = jax.random.randint(key, (), 0, len(self.consts.SCORPION_Y_POSITIONS))
        y = self.consts.SCORPION_Y_POSITIONS[idx_y]

        key_speed, _ = jax.random.split(key, 2)
        speed = jax.lax.cond(
            score >= 20000,
            lambda: jnp.where(jax.random.uniform(key_speed) < 0.75, 2, 1),
            lambda: 1
        )

        return jnp.stack([x, y, direction, speed])

    ## -------- Centipede Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def initialize_centipede_positions(self, wave: chex.Array) -> chex.Array:
        base_x = 80
        base_y = 5

        wave = wave[0]
        slow_wave = wave < 0
        num_heads = jnp.abs(wave)
        main_segments = self.consts.MAX_SEGMENTS - num_heads

        def spawn_segment(i):
            def main_body():
                is_head = i == 0
                return jnp.where(
                    slow_wave,
                    jnp.where(
                        is_head,
                        jnp.array([base_x + 4 * i, base_y, -1, 1, 2]),
                        jnp.array([base_x + 4 * i, base_y, -1, 1, 1]),
                    ),
                    jnp.where(
                        is_head,
                        jnp.array([base_x + 4 * i, base_y, -2, 1, 2]),
                        jnp.array([base_x + 4 * i, base_y, -2, 1, 1]),
                    )
                )

            def single_head():      # May not be 100% accurate (1-2px offset, varying per round)
                j = i - main_segments
                return jnp.where(
                    j == 0,
                    jnp.array([140, 5, -2, 1, 2]),
                    jnp.where(
                        j == 1,
                        jnp.array([16, 5, 2, 1, 2]),
                        jnp.where(
                            j == 2,
                            jnp.array([108, 5, -2, 1, 2]),
                            jnp.where(
                                j == 3,
                                jnp.array([48, 14, 2, 1, 2]),
                                jnp.where(
                                    j == 4,
                                    jnp.array([124, 23, 2, 1, 2]),
                                    jnp.where(
                                        j == 5,
                                        jnp.array([32, 14, 2, 1, 2]),
                                        jnp.where(
                                            j == 6,
                                            jnp.array([92, 14, -2, 1, 2]),
                                            jnp.where(
                                                j == 7,
                                                jnp.array([64, 14, -2, 1, 2]),
                                                jnp.array([80, 5, -2, 1, 2]),       # failsafe
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                )

            return jax.lax.cond(
                i < main_segments,
                main_body,
                single_head,
            )

        carry = jnp.arange(0, 9)
        return jax.vmap(spawn_segment)(carry).astype(jnp.float32)

    ## -------- Wave Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def process_wave(
            self,
            centipede_state: chex.Array,
            wave: chex.Array,
            score: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        logical_wave = wave[0]
        ui_wave = wave[1]

        def new_wave():
            new_logical_wave = jnp.where(
                score < 40_000,
                jnp.where(
                    logical_wave < 0,
                    logical_wave * -1 % 8,
                    (logical_wave + 1) * -1 % -8
                ),
                jnp.abs(logical_wave) + 1 % 8
            )

            new_wave = jnp.array([new_logical_wave, ui_wave + 1])
            return self.initialize_centipede_positions(new_wave), new_wave

        return jax.lax.cond(
            jnp.sum(centipede_state[:, 4]) == 0,
            lambda: new_wave(),
            lambda: (centipede_state, wave)
        )

    ## -------- Player Move Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def player_step(
            self,
            player_x: chex.Array,
            player_y: chex.Array,
            player_velocity_x: chex.Array,
            action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        up = jnp.isin(action, jnp.array([
            Action.UP,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.UPFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE
        ]))
        down = jnp.isin(action, jnp.array([
            Action.DOWN,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.DOWNFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]))
        left = jnp.isin(action, jnp.array([
            Action.LEFT,
            Action.UPLEFT,
            Action.DOWNLEFT,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNLEFTFIRE
        ]))
        right = jnp.isin(action, jnp.array([
            Action.RIGHT,
            Action.UPRIGHT,
            Action.DOWNRIGHT,
            Action.RIGHTFIRE,
            Action.UPRIGHTFIRE,
            Action.DOWNRIGHTFIRE
        ]))

        # x acceleration
        acc_dir = jnp.where(right, 1, jnp.where(left, -1, 0))
        no_horiz_op = jnp.invert(jnp.logical_or(right, left))
        turn_around = jnp.logical_or(
            jnp.logical_and(jnp.greater(player_velocity_x, 1/32), left),
            jnp.logical_and(jnp.less(player_velocity_x, -1/32), right),
        )

        raw_vel_x = jnp.fix(player_velocity_x)
        new_velocity_x = jnp.where(
            no_horiz_op,
            jnp.where(
                player_x % 4 == 0,
                0,
                1/32 * jnp.sign(player_velocity_x),
            ),
            jnp.where(
                turn_around,
                1/32 * jnp.sign(player_velocity_x),
                jnp.clip(
                    jnp.where(jnp.abs(raw_vel_x) * 2 < 1, player_velocity_x + 1/4 * acc_dir, player_velocity_x + 1/8 * acc_dir),
                    -3, 3,
                ),
            )
        )
        new_player_x = jnp.where(
            jnp.logical_and(no_horiz_op, player_x % 4 != 0),
            player_x + jnp.where(new_velocity_x < 0, -1, 1),
            jnp.clip(player_x + raw_vel_x, self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[0][1])
        ).astype(jnp.int32)

        # Calculate new y position
        y_idx = jnp.argmax(self.consts.PLAYER_Y_VALUES == player_y)
        new_idx = jnp.clip(
            y_idx + jnp.where(up, -1, jnp.where(down, 1, 0)),
            0, self.consts.PLAYER_Y_VALUES.shape[0] - 1)
        new_player_y = self.consts.PLAYER_Y_VALUES[new_idx]

        return new_player_x, new_player_y, new_velocity_x

    ## -------- Player Spell Logic -------- ##
    def player_spell_step(
            self,
            player_x: chex.Array,
            player_y: chex.Array,
            player_spell: chex.Array,
            action: chex.Array
    ) -> chex.Array:

        fire = jnp.isin(action, jnp.array([
            Action.FIRE,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]))

        spawn = jnp.logical_and(jnp.logical_not(player_spell[2] != 0), fire)

        new_is_alive = jnp.where(
            spawn,  # on spawn
            1,
            jnp.where(
                player_spell[1] < 0,
                0,
                player_spell[2]
            )  # on kill or keep
        )
        new_x = jnp.where(
            spawn,
            player_x + 1,
            jnp.where(new_is_alive, player_spell[0], 0)
        )
        new_y = jnp.where(
            spawn,
            jnp.floor(player_y) - 9,
            jnp.where(new_is_alive, player_spell[1] - self.consts.PLAYER_SPELL_SPEED, 0)
        )

        return jnp.array([new_x, new_y, new_is_alive])

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key = 42) -> Tuple[CentipedeObservation, CentipedeState]:
        """Initialize game state"""

        key = jax.random.PRNGKey(time.time_ns() % (2 ** 32))  # Pseudo random number generator seed key, based on current system time.
        new_key0, key_spider, key_scorpion = jax.random.split(key, 3)

        initial_spider_timer = jax.random.randint(key_spider, (), self.consts.SPIDER_MIN_SPAWN_FRAMES, self.consts.SPIDER_MAX_SPAWN_FRAMES + 1)
        initial_scorpion_timer = jax.random.randint(key_scorpion, (), self.consts.SCORPION_MIN_SPAWN_FRAMES, self.consts.SCORPION_MAX_SPAWN_FRAMES + 1)

        reset_state = CentipedeState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_velocity_x=jnp.array(0.0),
            player_spell=jnp.zeros(3, dtype=jnp.int32),
            mushroom_positions=self.initialize_mushroom_positions(),
            centipede_position=self.initialize_centipede_positions(jnp.array([0, 0])),
            centipede_spawn_timer=jnp.array(0),
            spider_position=jnp.zeros(3, dtype=jnp.int32),
            spider_spawn_timer=initial_spider_timer,
            spider_points=jnp.array([0, 0]),
            flea_position=jnp.zeros(3),
            flea_spawn_timer=jnp.array(0),
            scorpion_position=jnp.zeros(4, dtype=jnp.int32),
            scorpion_spawn_timer=initial_scorpion_timer,
            score=jnp.array(0),
            lives=jnp.array(3),
            step_counter=jnp.array(0),
            wave=jnp.array([0, 0]),
            death_counter=jnp.array(0),
            spark_position=jnp.zeros(2, dtype=jnp.int32),
            rng_key=new_key0,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CentipedeState, action: chex.Array) -> tuple[
        CentipedeObservation, CentipedeState, float, bool, CentipedeInfo]:
        # Translate compact agent action index to ALE console action
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        previous_state = state  # for reward/info

        def handle_death_animation():

            def get_mushroom_for_spark(death_counter, mushroom_positions):
                # Optimized: No sorting! Just find the Nth alive mushroom in array order
                # This is MUCH faster than sorting 304 mushrooms twice + looping
                mush_alive = jnp.count_nonzero(mushroom_positions[:, 3])
                alive_idx = (mush_alive - jnp.ceil(death_counter / 4)).astype(jnp.int32)
                
                # Mask for alive mushrooms
                alive_mask = mushroom_positions[:, 3] > 0
                
                # Cumulative count of alive mushrooms (vectorized, no loop!)
                alive_count = jnp.cumsum(alive_mask.astype(jnp.int32))
                
                # Find the alive_idx-th alive mushroom (where count equals alive_idx + 1)
                target_mask = (alive_count == (alive_idx + 1)) & alive_mask
                
                # Get first matching index
                target_idx = jnp.argmax(target_mask.astype(jnp.int32))
                
                # Return position or [0,0] if none found
                return jnp.where(
                    jnp.any(target_mask) & (alive_idx >= 0),
                    mushroom_positions[target_idx, :2].astype(jnp.int32),
                    jnp.zeros(2, dtype=jnp.int32)
                )

            def soft_reset():
                 new_key, key_spider, key_scorpion = jax.random.split(state.rng_key, 3)
                 initial_spider_timer = jax.random.randint(
                     key_spider,
                     (),
                     self.consts.SPIDER_MIN_SPAWN_FRAMES,
                     self.consts.SPIDER_MAX_SPAWN_FRAMES + 1
                 )
                 initial_scorpion_timer = jax.random.randint(
                     key_scorpion,
                     (),
                     self.consts.SCORPION_MIN_SPAWN_FRAMES,
                     self.consts.SCORPION_MAX_SPAWN_FRAMES + 1
                 )

                 return state.replace(
                     player_x=jnp.array(self.consts.PLAYER_START_X),
                     player_y=jnp.array(self.consts.PLAYER_START_Y),
                     player_velocity_x=jnp.array(0.0),
                     player_spell=jnp.zeros(3, dtype=jnp.int32),
                     mushroom_positions=state.mushroom_positions,
                     centipede_position=self.initialize_centipede_positions(jnp.array([0, 0])),
                     centipede_spawn_timer=jnp.array(0),
                     spider_position=jnp.zeros(3, dtype=jnp.int32),
                     spider_spawn_timer=initial_spider_timer,
                     spider_points=jnp.array([0, 0]),
                     flea_position=jnp.zeros(3),
                     flea_spawn_timer=jnp.array(0),
                     scorpion_position=jnp.zeros(4, dtype=jnp.int32),
                     scorpion_spawn_timer=initial_scorpion_timer,
                     score=state.score,
                     lives=state.lives - 1,
                     step_counter=state.step_counter,
                     wave=state.wave,
                     death_counter=jnp.array(0),
                     spark_position=jnp.zeros(2, dtype=jnp.int32),
                     rng_key=new_key,
                 )

            def compute_mushroom_frames():
                mush_alive = jnp.count_nonzero(state.mushroom_positions[:, 3])
                return mush_alive * 4

            new_death_counter = jax.lax.cond(
                state.death_counter <= -self.consts.DEATH_ANIMATION_MUSHROOM_THRESHOLD,
                lambda: compute_mushroom_frames(),
                lambda: (state.death_counter - 1),
            )

            new_score = jnp.where(
                jnp.logical_and(new_death_counter > 0, new_death_counter % 8 == 0),
                state.score + 5,
                state.score
            )

            new_spark_position = jax.lax.cond(
                jnp.logical_and(state.death_counter > 0, (state.death_counter - 1) % 4 >= 2),
                lambda: get_mushroom_for_spark(state.death_counter, state.mushroom_positions),
                lambda: jnp.zeros(2, dtype=jnp.int32)
            )

            state_during_animation = state.replace(
                player_spell=jnp.zeros(3, dtype=jnp.int32),
                spider_position=jnp.zeros(3, dtype=jnp.int32),
                spider_points=jnp.array([0, 0]),
                centipede_position=jnp.zeros_like(state.centipede_position),
                scorpion_position=jnp.zeros(4, dtype=jnp.int32),
                flea_position=jnp.zeros(3),
                death_counter=new_death_counter,
                score=new_score,
                spark_position=new_spark_position,
            )

            return jax.lax.cond(
                new_death_counter == 0,
                lambda: soft_reset(),
                lambda: state_during_animation
            )

        def normal_game_step():
            new_death_counter = self.check_player_enemy_collision(
                state.player_x,
                state.player_y,
                state.centipede_position,
                state.spider_position,
                state.flea_position
            )

            # --- Player Movement ---
            new_player_x, new_player_y, new_velocity_x = self.player_step(
                state.player_x, state.player_y, state.player_velocity_x, action
            )

            new_player_spell_state = self.player_spell_step(
                new_player_x, new_player_y, state.player_spell, action
            )

            # --- Mushroom Collision ---
            new_player_spell_state, updated_mushrooms, new_score = self.check_spell_mushroom_collision(
                new_player_spell_state, state.mushroom_positions, state.score
            )

            # --- Centipede Collision ---
            new_player_spell_state, new_centipede_position, updated_mushrooms, new_score = \
                self.check_spell_centipede_collision(
                    new_player_spell_state, state.centipede_position, updated_mushrooms, new_score
                )

            # --- Spider collision with mushrooms ---
            updated_mushrooms = self.check_spider_mushroom_collision(
                state.spider_position,
                updated_mushrooms
            )

            # --- Centipede Step & Wave ---
            new_centipede_position = self.centipede_step(new_centipede_position, state.mushroom_positions)
            new_centipede_position, new_wave = self.process_wave(new_centipede_position, state.wave, state.score)

            # --- Centipede Head Spawn Timer ---
            new_centipede_timer, new_centipede_position = self.handle_centipede_segment_spawn(      # Spawn new heads once centipede has reached bottom of screen
                state.centipede_spawn_timer,
                new_centipede_position
            )

            # --- Spider Collision ---
            new_player_spell_state, new_spider_position, new_score, new_spider_points_pre = self.check_spell_spider_collision(
                new_player_spell_state,
                state.spider_position,
                new_score,
                new_player_y,
                state.spider_points,
            )

            # --- Flea Collision ---
            new_player_spell_state, new_flea_position, new_flea_timer, new_score = self.check_spell_flea_collision(
                new_player_spell_state,
                state.flea_position,
                state.flea_spawn_timer,
                new_score
            )

            # --- Scorpion Collision ---
            new_player_spell_state, scorpion_after_hit, new_score, poison_stop_flag = self.check_spell_scorpion_collision(
                new_player_spell_state,
                state.scorpion_position,
                new_score
            )

            # --- Create new keys for next frame ---
            (
                new_rng_key,
                spider_spawn_key,
                spider_step_key,
                flea_rng_key,
                scorpion_spawn_key,
                scorpion_step_key
            ) = jax.random.split(
                state.rng_key,
                6
            )

            # --- Spider Movement & Spawn Timer ---
            spider_x, spider_y, spider_dir = new_spider_position
            spider_alive = spider_dir != 0

            new_spider_position, new_spider_timer = jax.lax.cond(
                spider_alive,
                lambda _: self.spider_alive_step(
                    spider_x,
                    spider_y,
                    spider_dir,
                    state.step_counter,
                    spider_step_key,
                    state.spider_spawn_timer
                ),
                lambda _: self.spider_dead_step(
                    state.spider_position,
                    state.spider_spawn_timer,
                    spider_spawn_key),
                operand=None
            )

            # --- Flea Handling
            new_flea_position, new_flea_timer, updated_mushrooms = self.flea_step(
                new_flea_position,
                new_flea_timer,
                updated_mushrooms,
                state.wave,
                flea_rng_key,
            )

            # --- Scorpion Movement & Spawn Timer ---
            scorpion_x, scorpion_y, scorpion_dir, scorpion_speed = scorpion_after_hit
            scorpion_alive = scorpion_dir != 0

            new_scorpion_position, new_scorpion_timer, updated_mushrooms = jax.lax.cond(
                scorpion_alive,
                lambda _: self.scorpion_alive_step(
                    scorpion_x,
                    scorpion_y,
                    scorpion_dir,
                    scorpion_speed,
                    scorpion_step_key,
                    state.scorpion_spawn_timer,
                    updated_mushrooms,
                    poison_stop_flag
                ),
                lambda _: self.scorpion_dead_step(
                    scorpion_after_hit,
                    state.scorpion_spawn_timer,
                    scorpion_spawn_key,
                    new_wave,
                    updated_mushrooms,
                    state.score,
                ),
                operand=None
            )

            # --- Spider points ---
            new_spider_points = jnp.where(
                new_spider_points_pre[1] > 0,
                jnp.array([new_spider_points_pre[0], new_spider_points_pre[1] - 1]),
                new_spider_points_pre
            )

            # Additional Life Every 10.000 points
            new_lives = jnp.where(
                jnp.logical_or(
                    new_score // 10000 == state.score // 10000,
                    state.lives >= 6
                ),
                state.lives,
                state.lives + 1
            )

            # --- New wave ---
            new_centipede_timer = jnp.where(state.wave[0] == new_wave[0], new_centipede_timer, 0)

            # --- Return State ---
            return state.replace(
                player_x=new_player_x,
                player_y=new_player_y,
                player_velocity_x=new_velocity_x,
                player_spell=new_player_spell_state,
                mushroom_positions=updated_mushrooms,
                centipede_position=new_centipede_position,
                centipede_spawn_timer=new_centipede_timer,
                spider_position=new_spider_position,
                spider_spawn_timer=new_spider_timer,
                spider_points=new_spider_points,
                scorpion_position=new_scorpion_position,
                scorpion_spawn_timer=new_scorpion_timer,
                flea_position=new_flea_position,
                flea_spawn_timer=new_flea_timer,
                score=jnp.where(new_score <= 999999, new_score, state.score),
                lives=new_lives,
                step_counter=state.step_counter + 1,
                wave=new_wave,
                death_counter=new_death_counter,
                spark_position=jnp.zeros(2, dtype=jnp.int32),
                rng_key=new_rng_key
            )

        normal_step_state = normal_game_step()
        death_animation_state = handle_death_animation()

        return_state = jax.lax.cond(
            state.lives == 0,       # If no more lives
            lambda: state,
            lambda: jax.lax.cond(
                state.death_counter == 0,       # If not dead
                lambda: normal_step_state,
                lambda: death_animation_state,
            )
        )

        obs = self._get_observation(return_state)
        done = self._get_done(return_state)
        env_reward = self._get_reward(previous_state, return_state)
        info = self._get_info(return_state)

        return obs, return_state, env_reward, done, info


class CentipedeRenderer(JAXGameRenderer):
    def __init__(self, consts: CentipedeConstants = None):
        super().__init__()
        self.consts = consts or CentipedeConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/centipede"

        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
            #downscale=(84, 84)
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

        # 4. Store original sprite color IDs (for palette swapping)
        # These are based on the first frame/color in the original game
        # Convert JAX arrays to numpy arrays, then to tuples for dictionary lookup
        self.PLAYER_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0)
        self.CENTIPEDE_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0)
        self.MUSHROOM_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0)
        self.SPIDER_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PURPLE)), 0)
        self.FLEA_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0)
        self.SCORPION_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0)
        self.BORDER_ORIGINAL_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0)

        # We assume the 4 spark sprites all use the same color in their source .npy
        # and that we will dynamically "paint" them. Let's assume their original
        # color is white (or the first color in the palette).
        # We find this by checking the color of the 'sparks' sprite.
        # This is complex. Let's assume they are different colors as in the old code.
        # We need IDs for Blue, Yellow, Red, Orange for the sparks.
        self.SPARK_BLUE_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0)
        self.SPARK_YELLOW_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.YELLOW)), 0)
        self.SPARK_RED_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0)
        self.SPARK_ORANGE_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0)

        # 5. Build color ID lookup tables for waves
        self.PLAYER_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.YELLOW)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PURPLE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0),
        ])
        self.CENTIPEDE_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PURPLE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.YELLOW)), 0),
        ])
        # (Mushrooms share colors with Player)
        self.SPIDER_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PURPLE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.YELLOW)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),
        ])
        self.FLEA_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.YELLOW)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PURPLE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0),
        ])
        self.SCORPION_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.YELLOW)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PURPLE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0),
        ])
        self.BORDER_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.GREEN)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.LIGHT_BLUE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_PURPLE)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.DARK_YELLOW)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.PINK)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),
            self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.RED)), 0),  # Wave 7 also uses RED
        ])

        # 6. Replicate Animation Stack Logic (from old load_sprites)
        self.SHAPE_MASKS['poisoned_mushroom'] = jnp.concatenate(
            [jnp.repeat(self.SHAPE_MASKS['poisoned_mushroom_group'][i][None], 4, axis=0) for i in range(16)]
        )
        self.SHAPE_MASKS['spider'] = jnp.concatenate(
            [jnp.repeat(self.SHAPE_MASKS['spider_group'][i][None], 8, axis=0) for i in range(4)]
        )
        self.SHAPE_MASKS['flea'] = jnp.concatenate(
            [jnp.repeat(self.SHAPE_MASKS['flea_group'][i][None], 2, axis=0) for i in range(2)]
        )
        self.SHAPE_MASKS['scorpion'] = jnp.concatenate(
            [jnp.repeat(self.SHAPE_MASKS['scorpion_group'][i][None], 8, axis=0) for i in range(2)]
        )

        # 7. Store animation lengths
        self.anim_len = {
            'poisoned_mushroom': self.SHAPE_MASKS['poisoned_mushroom'].shape[0],  # 64
            'spider': self.SHAPE_MASKS['spider'].shape[0],  # 32
            'flea': self.SHAPE_MASKS['flea'].shape[0],  # 4
            'scorpion': self.SHAPE_MASKS['scorpion'].shape[0],  # 16
            'sparks': self.SHAPE_MASKS['sparks'].shape[0],  # 4
        }

        # 8. Pre-build colored spark masks
        PLACEHOLDER_ID = self.COLOR_TO_ID.get(tuple(np.asarray(self.consts.ORANGE)), 1)
        spark_colors = jnp.array([
            self.SPARK_BLUE_ID, self.SPARK_YELLOW_ID, self.SPARK_RED_ID, self.SPARK_ORANGE_ID
        ])

        def create_colored_mask(spark_mask, color_id):
            return jnp.where(spark_mask == PLACEHOLDER_ID, color_id, spark_mask)

        self.COLORED_SPARK_MASKS = jax.vmap(create_colored_mask)(self.SHAPE_MASKS['sparks'], spark_colors)

    @partial(jax.jit, static_argnums=(0,))
    def _get_frame_palette(self, wave: chex.Array) -> chex.Array:
        """Dynamically build the palette for the current wave."""
        wave_mod = wave[1] % 8
        frame_palette = self.PALETTE

        frame_palette = frame_palette.at[self.PLAYER_ORIGINAL_ID].set(self.PLAYER_COLOR_IDS[wave_mod])
        frame_palette = frame_palette.at[self.CENTIPEDE_ORIGINAL_ID].set(self.CENTIPEDE_COLOR_IDS[wave_mod])
        frame_palette = frame_palette.at[self.MUSHROOM_ORIGINAL_ID].set(self.PLAYER_COLOR_IDS[wave_mod])  # Uses player color
        frame_palette = frame_palette.at[self.SPIDER_ORIGINAL_ID].set(self.SPIDER_COLOR_IDS[wave_mod])
        frame_palette = frame_palette.at[self.FLEA_ORIGINAL_ID].set(self.FLEA_COLOR_IDS[wave_mod])
        frame_palette = frame_palette.at[self.SCORPION_ORIGINAL_ID].set(self.SCORPION_COLOR_IDS[wave_mod])
        frame_palette = frame_palette.at[self.BORDER_ORIGINAL_ID].set(self.BORDER_COLOR_IDS[wave_mod])

        # Spider score colors are static and do not need to be swapped
        return frame_palette

    @partial(jax.jit, static_argnums=(0,))
    def _render_sparks(self, raster, state: CentipedeState) -> chex.Array:
        """Render player death or mushroom healing sparks."""

        def no_render(r):
            return r

        def player_sparks(r):
            # Sparks logic is complex, just select a frame
            idx = (-(state.death_counter // 4 + 1)) % 4
            spark_mask = self.COLORED_SPARK_MASKS[idx]

            return self.jr.render_at_clipped(
                r, state.player_x - 4, state.player_y + 3,
                spark_mask, flip_offset=self.FLIP_OFFSETS['sparks']
            )

        def mushroom_sparks(r):
            idx = (-jnp.mod(state.death_counter, 4)).astype(jnp.int32)
            spark_mask = self.COLORED_SPARK_MASKS[idx]


            mush_pos = state.spark_position

            return self.jr.render_at_clipped(
                r, mush_pos[0] - 2, mush_pos[1] - 2, # Use the pre-calculated pos
                spark_mask, flip_offset=self.FLIP_OFFSETS['sparks']
            )

        return jax.lax.cond(
            state.death_counter != 0,
            lambda r: jax.lax.cond(
                state.death_counter < 0,
                player_sparks,
                lambda r_in: jax.lax.cond(
                    (state.death_counter - 1) % 4 >= 2,
                    mushroom_sparks,
                    no_render,
                    r_in
                ),
                r
            ),
            no_render,
            raster
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CentipedeState):
        # --- 1. Setup ---
        # Start with the static background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # --- 2. Get Dynamic Animation Frames ---
        poison_anim_idx = state.step_counter % self.anim_len['poisoned_mushroom']
        spider_anim_idx = state.step_counter % self.anim_len['spider']
        flea_anim_idx = state.step_counter % self.anim_len['flea']
        scorpion_anim_idx = state.step_counter % self.anim_len['scorpion']

        # --- 3. Render player ---
        raster = jax.lax.cond(
            state.death_counter >= 0,
            lambda r: self.jr.render_at(
                r, state.player_x, state.player_y,
                self.SHAPE_MASKS['player'], flip_offset=self.FLIP_OFFSETS['player']
            ),
            lambda r: r,
            raster
        )

        # --- 4. Render player spell ---
        raster = jax.lax.cond(
            state.player_spell[2] != 0,
            lambda r: self.jr.render_at_clipped(
                r, state.player_spell[0], state.player_spell[1],
                self.SHAPE_MASKS['player_spell'], flip_offset=self.FLIP_OFFSETS['player_spell']
            ),
            lambda r: r,
            raster
        )

        # --- 5. Render mushrooms ---
        frame_mushroom = self.SHAPE_MASKS['mushroom']
        frame_poisoned_mushroom = self.SHAPE_MASKS['poisoned_mushroom'][poison_anim_idx]
        mush_offset = self.FLIP_OFFSETS['mushroom']
        pmush_offset = self.FLIP_OFFSETS['poisoned_mushroom_group']  # Assumes all padded same

        def render_mushroom_scan(raster_in, pos):
            x, y, poisoned, lives = pos
            alive = lives > 0
            is_poisoned = poisoned == 1

            def render_fn(r):
                mask = jax.lax.select(is_poisoned, frame_poisoned_mushroom, frame_mushroom)
                offset = jax.lax.select(is_poisoned, pmush_offset, mush_offset)
                return self.jr.render_at(r, x, y, mask, flip_offset=offset)

            return jax.lax.cond(alive, render_fn, lambda r: r, raster_in), None

        raster, _ = jax.lax.scan(render_mushroom_scan, raster, state.mushroom_positions)

        # --- 6. Render centipede ---
        frame_centipede = self.SHAPE_MASKS['centipede']
        cent_offset = self.FLIP_OFFSETS['centipede']

        def render_segment_scan(raster_in, pos):
            x, y, _, _, alive_flag = pos
            should_render = (alive_flag != 0) & (state.death_counter <= 0)

            def render_fn(r):
                return self.jr.render_at(r, x, y, frame_centipede, flip_offset=cent_offset)

            return jax.lax.cond(should_render, render_fn, lambda r: r, raster_in), None

        raster, _ = jax.lax.scan(render_segment_scan, raster, state.centipede_position)

        # --- 7. Render spider ---
        frame_spider = self.SHAPE_MASKS['spider'][spider_anim_idx]
        spider_offset = self.FLIP_OFFSETS['spider_group']

        raster = jax.lax.cond(
            state.spider_position[2] != 0,
            lambda r: self.jr.render_at_clipped(
                r, state.spider_position[0] + 2, state.spider_position[1] - 2,
                frame_spider, flip_offset=spider_offset
            ),
            lambda r: r,
            raster
        )

        # --- 8. Render spider score ---
        frame_spider300 = self.SHAPE_MASKS['spider_300']
        frame_spider600 = self.SHAPE_MASKS['spider_600']
        frame_spider900 = self.SHAPE_MASKS['spider_900']

        score_mask = jnp.where(
            state.spider_points[0] == 1, frame_spider300,
            jnp.where(state.spider_points[0] == 2, frame_spider600, frame_spider900)
        )
        # Assuming offsets are similar, pick one
        score_offset = self.FLIP_OFFSETS['spider_300']

        raster = jax.lax.cond(
            state.spider_points[1] != 0,
            lambda r: self.jr.render_at_clipped(
                r, state.spider_position[0] + 2, state.spider_position[1] - 2,
                score_mask, flip_offset=score_offset
            ),
            lambda r: r,
            raster
        )

        # --- 9. Render Flea ---
        frame_flea = self.SHAPE_MASKS['flea'][flea_anim_idx]
        flea_offset = self.FLIP_OFFSETS['flea_group']
        raster = jax.lax.cond(
            state.flea_position[2] != 0,
            lambda r: self.jr.render_at_clipped(
                r, state.flea_position[0], state.flea_position[1],
                frame_flea, flip_offset=flea_offset
            ),
            lambda r: r,
            raster
        )

        # --- 10. Render Scorpion ---
        frame_scorpion = self.SHAPE_MASKS['scorpion'][scorpion_anim_idx]
        scorpion_offset = self.FLIP_OFFSETS['scorpion_group']

        raster = jax.lax.cond(
            state.scorpion_position[2] != 0,
            lambda r: self.jr.render_at_clipped(
                r, state.scorpion_position[0] + 2, state.scorpion_position[1] - 2,
                frame_scorpion,
                flip_horizontal=(state.scorpion_position[2] == -1),
                flip_offset=scorpion_offset
            ),
            lambda r: r,
            raster
        )

        # --- 11. Render sparks ---
        # This function is complex and dynamically paints masks
        raster = self._render_sparks(raster, state)

        # --- 12. Render bottom border ---
        raster = self.jr.render_at(
            raster, 16, 183,
            self.SHAPE_MASKS['bottom_border'],
            flip_offset=self.FLIP_OFFSETS['bottom_border']
        )

        # --- 13. Render score ---
        score_array = self.jr.int_to_digits(state.score, max_digits=6)
        raster = self.jr.render_label(
            raster, 100, 187, score_array,
            self.SHAPE_MASKS['digits'], spacing=8, max_digits=6
        )

        # --- 14. Render life indicator ---
        raster = self.jr.render_indicator(
            raster, 16, 187, state.lives - 1,
            self.SHAPE_MASKS['life_indicator'],
            spacing=8, max_value=5  # Assuming max 5 lives
        )

        # --- 15. Final Palette Lookup (Now with Dynamic Updates) ---

        # Gather the palette indices we need to update
        wave_mod = state.wave[1] % 8
        indices_to_update = jnp.array([
            self.PLAYER_ORIGINAL_ID,
            self.CENTIPEDE_ORIGINAL_ID,
            self.MUSHROOM_ORIGINAL_ID,
            self.SPIDER_ORIGINAL_ID,
            self.FLEA_ORIGINAL_ID,
            self.SCORPION_ORIGINAL_ID,
            self.BORDER_ORIGINAL_ID
        ])

        # Gather the new color IDs for the current wave (i.e. which sprite type has which color now)
        new_color_ids = jnp.array([
            self.PLAYER_COLOR_IDS[wave_mod],
            self.CENTIPEDE_COLOR_IDS[wave_mod],
            self.PLAYER_COLOR_IDS[wave_mod],  # Mushroom uses player color
            self.SPIDER_COLOR_IDS[wave_mod],
            self.FLEA_COLOR_IDS[wave_mod],
            self.SCORPION_COLOR_IDS[wave_mod],
            self.BORDER_COLOR_IDS[wave_mod]
        ])

        return self.jr.render_from_palette(
            raster,
            self.PALETTE,
            indices_to_update=indices_to_update,
            new_color_ids=new_color_ids
        )