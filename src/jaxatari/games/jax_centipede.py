"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""

import os
import jax
import jax.numpy as jnp
import chex

import jaxatari.rendering.jax_rendering_utils as jru
import time
from functools import partial
from typing import NamedTuple, Tuple

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer


#from jaxatari.rendering.jax_rendering_utils import recolor_sprite


class CentipedeConstants:
    # -------- Game constants --------
    WIDTH = 160
    HEIGHT = 210
    SCALING_FACTOR = 6

    ## -------- Player constants --------
    PLAYER_START_X = 76
    PLAYER_START_Y = 172
    PLAYER_BOUNDS = (16, 140), (141, 172)

    PLAYER_SIZE = (4, 9)

    PLAYER_Y_VALUES = jnp.array([141, 145, 147, 150, 154, 156, 159, 163, 165, 168, 172])      # Double to not need extra state value

    ## -------- Player spell constants --------
    PLAYER_SPELL_SPEED = 9

    PLAYER_SPELL_SIZE = (1, 8)

    ## -------- Starting Pattern (X -> placed, O -> not placed, P -> placed and poisoned) --------
    MUSHROOM_STARTING_PATTERN = [
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
    ]

    ## -------- Mushroom constants --------
    MAX_MUSHROOMS = 304             # Default 304 (19*16) | Maximum number of mushrooms that can appear at the same time
    MUSHROOM_NUMBER_OF_ROWS = 19    # Default 19 | Number of rows -> Determines value of MAX_MUSHROOMS
    MUSHROOM_NUMBER_OF_COLS = 16    # Default 16 | Number of mushrooms per row -> Determines value of MAX_MUSHROOMS
    MUSHROOM_X_SPACING = 8      #
    MUSHROOM_Y_SPACING = 9
    MUSHROOM_COLUMN_START_EVEN = 20
    MUSHROOM_COLUMN_START_ODD = 16
    MUSHROOM_SIZE = (4, 3)
    MUSHROOM_HITBOX_Y_OFFSET = 6

    ## -------- Centipede constants --------
    MAX_SEGMENTS = 9
    SEGMENT_SIZE = (4, 6)

    ## -------- Spider constants --------
    SPIDER_X_POSITIONS = jnp.array([16, 133])
    SPIDER_Y_POSITIONS = jnp.array([115, 124, 133, 142, 151, 160, 169, 178])
    SPIDER_MOVE_PROBABILITY = 0.2
    SPIDER_MIN_SPAWN_FRAMES = 55
    SPIDER_MAX_SPAWN_FRAMES = 355
    SPIDER_SIZE = (8, 6)
    SPIDER_CLOSE_RANGE = 16
    SPIDER_MID_RANGE = 32

    ## -------- Scorpion constants --------
    SCORPION_X_POSITIONS = jnp.array([16, 133])
    SCORPION_Y_POSITIONS = jnp.array([7, 16, 25, 34, 43, 52, 61, 70, 79, 88, 97, 106, 115, 124, 133])
    SCORPION_MIN_SPAWN_FRAMES = 355 #TODO: prob. not accurate
    SCORPION_MAX_SPAWN_FRAMES = 2000 #TODO: prob. not accurate
    SCORPION_SIZE = (8, 6)
    SCORPION_POINTS = 1000

    ## -------- Flea constants --------
    FLEA_SIZE = (4, 6)
    FLEA_SPAWN_MUSHROOM_PROBABILITY = 0.5
    FLEA_POINTS = 200

    ## -------- Death animation constants --------
    DEATH_ANIMATION_MUSHROOM_THRESHOLD = 64        # 4 Frames * 4 Sprites * 4 Repetitions

    ## -------- Color constants --------
    ORANGE = jnp.array([181, 83, 40])#B55328    # Mushrooms lvl1
    DARK_ORANGE = jnp.array([198, 108, 58])#C66C3A
    PINK = jnp.array([184, 70, 162])#B846A2      # Centipede lvl1
    GREEN = jnp.array([110, 156, 66])#6E9C42     # Border lvl1
    LIGHT_PURPLE = jnp.array([188, 144, 252])#BC90FC  # UI Elements
    PURPLE = jnp.array([146, 70, 192])#9246C0        # Spider
    DARK_PURPLE = jnp.array([66, 72, 200])#4248C8
    LIGHT_BLUE = jnp.array([84, 138, 210])#548AD2    # Border lvl2
    DARK_BLUE = jnp.array([45, 50, 184])#2D32B8     # Mushrooms lvl2
    LIGHT_RED = jnp.array([200, 72, 72])#C84848
    RED = jnp.array([184, 50, 50])#B83232       # Centipede lvl2
    YELLOW = jnp.array([187, 187, 53])#BBBB35
    DARK_YELLOW = jnp.array([162, 162, 42])#A2A22A

    ## -------- Sprite Frames --------
    SPRITE_PLAYER_FRAMES = 1
    SPRITE_PLAYER_SPELL_FRAMES = 1
    SPRITE_CENTIPEDE_FRAMES = 1
    SPRITE_MUSHROOM_FRAMES = 1
    SPRITE_SPIDER_FRAMES = 4
    SPRITE_SPIDER_300_FRAMES = 1
    SPRITE_SPIDER_600_FRAMES = 1
    SPRITE_SPIDER_900_FRAMES = 1
    SPRITE_FLEA_FRAMES = 2
    SPRITE_SCORPION_FRAMES = 2
    SPRITE_SPARKS_FRAMES = 4
    SPRITE_BOTTOM_BORDER_FRAMES = 1
    SPRITE_POISONED_MUSHROOMS_FRAMES = 16

    # -------- Centipede States --------

class CentipedeState(NamedTuple):
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
    rng_key: chex.PRNGKey

class PlayerEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    o: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class CentipedeObservation(NamedTuple):
    player: PlayerEntity
    mushrooms: jnp.ndarray      # Shape (MAX_MUSHROOMS, 5) - MAX_MUSHROOMS mushrooms, each with x,y,w,h,active
    centipede: jnp.ndarray      # Shape (MAX_SEGMENTS, 5) - MAX_SEGMENTS Centipede segments, each with x,y,w,h,active
    spider: EntityPosition      # Shape (5,) - one spider with x,y,w,h,active
    flea: EntityPosition        # Shape (5,) - one flea with x,y,w,h,active
    scorpion: EntityPosition    # Shape (5,) - one scorpion with x,y,w,h,active
    player_spell: EntityPosition    # Shape (5,) - one spell with x,y,w,h,active
    score: jnp.ndarray
    lives: jnp.ndarray
    # TODO: fill
    # if changed: obs_to_flat_array, _get_observation, (step, reset)

class CentipedeInfo(NamedTuple):
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

# -------- Render Constants --------
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    player = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player/player.npy"))
    player_spell = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player_spell/player_spell.npy"))
    mushroom = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/mushrooms/mushroom.npy"))
    centipede = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/centipede/segment.npy"))
    spider1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/1.npy"))
    spider2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/2.npy"))
    spider3 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/3.npy"))
    spider4 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/4.npy"))
    spider_300 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider_scores/300.npy"))
    spider_600 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider_scores/600.npy"))
    spider_900 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider_scores/900.npy"))
    flea1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/flea/1.npy"))
    flea2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/flea/2.npy"))
    scorpion1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/scorpion/1.npy"))
    scorpion2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/scorpion/2.npy"))
    sparks1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/1.npy"))
    sparks2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/2.npy"))
    sparks3 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/3.npy"))
    sparks4 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/4.npy"))
    bottom_border = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/bottom_border.npy"))

    ## -------- poisoned mushrooms --------
    pMush1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/1.npy"))
    pMush2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/2.npy"))
    pMush3 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/3.npy"))
    pMush4 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/4.npy"))
    pMush5 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/5.npy"))
    pMush6 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/6.npy"))
    pMush7= jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/7.npy"))
    pMush8 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/8.npy"))
    pMush9 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/9.npy"))
    pMush10 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/10.npy"))
    pMush11 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/11.npy"))
    pMush12 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/12.npy"))
    pMush13 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/13.npy"))
    pMush14 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/14.npy"))
    pMush15 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/15.npy"))
    pMush16 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/poisoned_mushrooms/16.npy"))


    sparks_sprites, _ = jru.pad_to_match([sparks1, sparks2, sparks3, sparks4])
    spider_sprites, _ = jru.pad_to_match([spider1, spider2, spider3, spider4])
    flea_sprites, _ = jru.pad_to_match([flea1, flea2])
    scorpion_sprites, _ = jru.pad_to_match([scorpion1, scorpion2])
    poisoned_mushroom_sprites, _ = jru.pad_to_match([pMush1, pMush2, pMush3, pMush4,
                                                     pMush5, pMush6, pMush7, pMush8,
                                                     pMush9, pMush10, pMush11, pMush12,
                                                     pMush13, pMush14, pMush15, pMush16])

    SPRITE_PLAYER = jnp.expand_dims(player, 0)
    SPRITE_PLAYER_SPELL = jnp.expand_dims(player_spell, 0)

    SPRITE_CENTIPEDE = jnp.expand_dims(centipede, 0)
    SPRITE_MUSHROOM = jnp.expand_dims(mushroom, 0)

    SPRITE_POISONED_MUSHROOM = jnp.concatenate(
        [
            jnp.repeat(poisoned_mushroom_sprites[0][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[1][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[2][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[3][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[4][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[5][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[6][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[7][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[8][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[9][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[10][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[11][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[12][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[13][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[14][None], 4, axis=0),
            jnp.repeat(poisoned_mushroom_sprites[15][None], 4, axis=0),
        ]
    )

    SPRITE_SPIDER = jnp.concatenate(
        [
            jnp.repeat(spider_sprites[0][None], 8, axis=0),
            jnp.repeat(spider_sprites[1][None], 8, axis=0),
            jnp.repeat(spider_sprites[2][None], 8, axis=0),
            jnp.repeat(spider_sprites[3][None], 8, axis=0),
        ]
    )
    SPRITE_SPIDER_300 = jnp.expand_dims(spider_300, 0)
    SPRITE_SPIDER_600 = jnp.expand_dims(spider_600, 0)
    SPRITE_SPIDER_900 = jnp.expand_dims(spider_900, 0)
    SPRITE_FLEA = jnp.concatenate(
        [
            jnp.repeat(flea_sprites[0][None], 2, axis=0),
            jnp.repeat(flea_sprites[1][None], 2, axis=0),
        ]
    )
    SPRITE_SCORPION = jnp.concatenate(
        [
            jnp.repeat(scorpion_sprites[0][None], 8, axis=0),
            jnp.repeat(scorpion_sprites[1][None], 8, axis=0),
        ]
    )

    SPRITE_SPARKS = jnp.concatenate(
        [
            jnp.repeat(sparks_sprites[0][None], 1, axis=0),
            jnp.repeat(sparks_sprites[1][None], 1, axis=0),
            jnp.repeat(sparks_sprites[2][None], 1, axis=0),
            jnp.repeat(sparks_sprites[3][None], 1, axis=0),
        ]
    )
    SPRITE_BOTTOM_BORDER = jnp.expand_dims(bottom_border, 0)

    DIGITS = jru.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/centipede/big_numbers/{}.npy"))
    LIFE_INDICATOR = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/wand.npy"))

    return (
        SPRITE_PLAYER,
        SPRITE_PLAYER_SPELL,
        SPRITE_CENTIPEDE,
        SPRITE_MUSHROOM,
        SPRITE_POISONED_MUSHROOM,
        SPRITE_SPIDER,
        SPRITE_SPIDER_300,
        SPRITE_SPIDER_600,
        SPRITE_SPIDER_900,
        SPRITE_FLEA,
        SPRITE_SCORPION,
        SPRITE_SPARKS,
        SPRITE_BOTTOM_BORDER,
        DIGITS,
        LIFE_INDICATOR,
    )

(
    SPRITE_PLAYER,
    SPRITE_PLAYER_SPELL,
    SPRITE_CENTIPEDE,
    SPRITE_MUSHROOM,
    SPRITE_POISONED_MUSHROOM,
    SPRITE_SPIDER,
    SPRITE_SPIDER_300,
    SPRITE_SPIDER_600,
    SPRITE_SPIDER_900,
    SPRITE_FLEA,
    SPRITE_SCORPION,
    SPRITE_SPARKS,
    SPRITE_BOTTOM_BORDER,
    DIGITS,
    LIFE_INDICATOR,
) = load_sprites()

# -------- Game Logic --------

class JaxCentipede(JaxEnvironment[CentipedeState, CentipedeObservation, CentipedeInfo, CentipedeConstants]):
    def __init__(self, consts: CentipedeConstants = None, reward_funcs: list[callable] = None):
        consts = consts or CentipedeConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
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
            Action.DOWNLEFTFIRE
        ]
        self.frame_stack_size = 4
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
        return spaces.Discrete(len(self.action_set))

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
    def _get_info(self, state: CentipedeState, all_rewards: jnp.ndarray = None) -> CentipedeInfo:
        return CentipedeInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: CentipedeState, state: CentipedeState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: CentipedeState, state: CentipedeState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

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

        # --- Run update over all segments ---
        init_carry = jnp.zeros_like(centipede_state[:, 0], dtype=jnp.int32), False
        turn_around, _ = jax.lax.fori_loop(0, centipede_state.shape[0] - 1, lambda i, carry: carry, init_carry)

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
    def check_spell_centipede_collision(        # TODO: fix
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
        # Create row and column indices
        row_indices = jnp.repeat(
            jnp.arange(self.consts.MUSHROOM_NUMBER_OF_ROWS),
            self.consts.MUSHROOM_NUMBER_OF_COLS
        )
        col_indices = jnp.tile(
            jnp.arange(self.consts.MUSHROOM_NUMBER_OF_COLS),
            self.consts.MUSHROOM_NUMBER_OF_ROWS
        )

        # Compute row parity
        row_is_even = row_indices % 2 == 0
        column_start = jnp.where(
            row_is_even,
            self.consts.MUSHROOM_COLUMN_START_EVEN,
            self.consts.MUSHROOM_COLUMN_START_ODD
        )
        x = column_start + self.consts.MUSHROOM_X_SPACING * col_indices
        x = x.astype(jnp.int32)

        y = (row_indices * self.consts.MUSHROOM_Y_SPACING + 7).astype(jnp.int32)

        # Build full pattern as arrays
        # Lives: 3 for 'X' or 'P', 0 otherwise
        lives_pattern = jnp.array([
            [3 if c.upper() in ("X", "P") else 0 for c in row.ljust(self.consts.MUSHROOM_NUMBER_OF_COLS, "O")]
            for row in self.consts.MUSHROOM_STARTING_PATTERN
        ])

        # Poison: 1 for 'P', 0 otherwise
        poison_pattern = jnp.array([
            [1 if c.upper() == "P" else 0 for c in row.ljust(self.consts.MUSHROOM_NUMBER_OF_COLS, "O")]
            for row in self.consts.MUSHROOM_STARTING_PATTERN
        ])

        # Pad patterns to required size
        pad_rows = max(0, self.consts.MUSHROOM_NUMBER_OF_ROWS - lives_pattern.shape[0])
        lives_pattern = jnp.pad(lives_pattern, ((0, pad_rows), (0, 0)), constant_values=0)
        poison_pattern = jnp.pad(poison_pattern, ((0, pad_rows), (0, 0)), constant_values=0)

        lives = lives_pattern[row_indices, col_indices]
        is_poisoned = poison_pattern[row_indices, col_indices]

        return jnp.stack([x, y, is_poisoned, lives], axis=1)

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
        initial_positions = jnp.zeros((self.consts.MAX_SEGMENTS, 5))

        wave = wave[0]
        slow_wave = wave < 0
        num_heads = jnp.abs(wave)
        main_segments = self.consts.MAX_SEGMENTS - num_heads

        def spawn_segment(i, segments: chex.Array):
            def main_body():
                is_head = i == 0
                return segments.at[i].set(
                    jnp.where(
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
                )

            def single_head():      # May not be 100% accurate (1-2px offset, varying per round)
                j = i - main_segments
                return jnp.where(
                    j == 0,
                    segments.at[i].set(jnp.array([140, 5, -2, 1, 2])),      # TODO: sometimes starts with different direction
                    jnp.where(
                        j == 1,
                        segments.at[i].set(jnp.array([16, 5, 2, 1, 2])),
                        jnp.where(
                            j == 2,
                            segments.at[i].set(jnp.array([108, 5, -2, 1, 2])),
                            jnp.where(
                                j == 3,
                                segments.at[i].set(jnp.array([48, 14, 2, 1, 2])),
                                jnp.where(
                                    j == 4,
                                    segments.at[i].set(jnp.array([124, 23, 2, 1, 2])),
                                    jnp.where(
                                        j == 5,
                                        segments.at[i].set(jnp.array([32, 14, 2, 1, 2])),
                                        jnp.where(
                                            j == 6,
                                            segments.at[i].set(jnp.array([92, 14, -2, 1, 2])),
                                            jnp.where(
                                                j == 7,
                                                segments.at[i].set(jnp.array([64, 14, -2, 1, 2])),
                                                segments.at[i].set(jnp.array([80, 5, -2, 1, 2])),       # failsafe
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

        return jax.lax.fori_loop(0, self.consts.MAX_SEGMENTS, spawn_segment, initial_positions)

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

            new_wave = jnp.array([new_logical_wave, ui_wave + 1 % 8])
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
    def player_spell_step(      # TODO: fix behaviour for close objects (add cooldown)
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
            rng_key=new_key0,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CentipedeState, action: chex.Array) -> tuple[
        CentipedeObservation, CentipedeState, float, bool, CentipedeInfo]:

        previous_state = state  # for reward/info

        def handle_death_animation():

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

                return state._replace(
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

            state_during_animation = state._replace(
                player_spell=jnp.zeros(3, dtype=jnp.int32),
                spider_position=jnp.zeros(3, dtype=jnp.int32),
                centipede_position=jnp.zeros_like(state.centipede_position),
                scorpion_position=jnp.zeros(4, dtype=jnp.int32),
                flea_position=jnp.zeros(3),
                death_counter=new_death_counter,
                score=new_score
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
            return state._replace(
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
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, env_reward, done, info


class CentipedeRenderer(JAXGameRenderer):
    def __init__(self, consts: CentipedeConstants = None):
        super().__init__()
        self.consts = consts or CentipedeConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CentipedeState):
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        def recolor_sprite(
                sprite: chex.Array,
                color: chex.Array,  # RGB, up to 4 dimensions
                bounds: tuple[int, int, int, int] = None  # (top, left, bottom, right)
        ) -> chex.Array:
            # Ensure color is the same dtype as sprite
            dtype = sprite.dtype
            color = color.astype(dtype)

            assert sprite.ndim == 3 and sprite.shape[2] in (3, 4), "Sprite must be HxWx3 or HxWx4"

            if color.shape[0] < sprite.shape[2]:
                missing = sprite.shape[2] - color.shape[0]
                pad = jnp.full((missing,), 255, dtype=dtype)
                color = jnp.concatenate([color, pad], axis=0)

            assert color.shape[0] == sprite.shape[2], "Color channels must match sprite channels"

            H, W, _ = sprite.shape

            if bounds is None:
                region = sprite
            else:
                top, left, bottom, right = bounds
                assert 0 <= left < right <= H and 0 <= top < bottom <= W, "Invalid bounds"
                region = sprite[left:right, top:bottom]

            visible_mask = jnp.any(region != 0, axis=-1, keepdims=True)  # (h, w, 1)

            color_broadcasted = jnp.broadcast_to(color, region.shape).astype(dtype)
            recolored_region = jnp.where(visible_mask, color_broadcasted, jnp.zeros_like(color_broadcasted))

            if bounds is None:
                return recolored_region
            else:
                recolored_sprite = sprite.at[left:right, top:bottom].set(recolored_region)
                return recolored_sprite

        def get_sprite_frames(wave: chex.Array, step_counter: int):
            """Gives all sprite-frames dynamically depending on step_counter and recoloring depending on wave"""

            def get_frame(sprite_id, num_frames: int, step_counter: int):
                """Calculates dynamically the frame of a sprite"""
                if num_frames == 1:
                    return jru.get_sprite_frame(sprite_id, 0)
                idx = step_counter
                return jru.get_sprite_frame(sprite_id, idx)

            def get_sparks(death_counter):
                # jax.debug.print("{}", jnp.mod(jnp.ceil(death_counter / 8), 4).astype(jnp.int32))
                sprites = jnp.array([
                    recolor_sprite(jru.get_sprite_frame(SPRITE_SPARKS, 0), self.consts.DARK_BLUE),
                    recolor_sprite(jru.get_sprite_frame(SPRITE_SPARKS, 1), self.consts.YELLOW),
                    recolor_sprite(jru.get_sprite_frame(SPRITE_SPARKS, 2), self.consts.RED),
                    recolor_sprite(jru.get_sprite_frame(SPRITE_SPARKS, 3), self.consts.ORANGE),
                ])
                return jnp.where(
                    death_counter < 0,
                    sprites[-(death_counter // 4 + 1) % 4],
                    sprites[-jnp.mod(death_counter, 4).astype(jnp.int32)],      # placeholder
                )


            # --- Get frames dynamically --- #
            frame_player_idx = get_frame(SPRITE_PLAYER, self.consts.SPRITE_PLAYER_FRAMES, step_counter)
            frame_player_spell_idx = get_frame(SPRITE_PLAYER_SPELL, self.consts.SPRITE_PLAYER_SPELL_FRAMES, step_counter)
            frame_centipede_idx = get_frame(SPRITE_CENTIPEDE, self.consts.SPRITE_CENTIPEDE_FRAMES, step_counter)
            frame_mushroom_idx = get_frame(SPRITE_MUSHROOM, self.consts.SPRITE_MUSHROOM_FRAMES, step_counter)
            frame_poisoned_mushroom_idx = get_frame(SPRITE_POISONED_MUSHROOM, self.consts.SPRITE_POISONED_MUSHROOMS_FRAMES, step_counter)
            frame_spider_idx = get_frame(SPRITE_SPIDER, self.consts.SPRITE_SPIDER_FRAMES, step_counter)
            frame_spider300_idx = get_frame(SPRITE_SPIDER_300, self.consts.SPRITE_SPIDER_300_FRAMES, step_counter)
            frame_spider600_idx = get_frame(SPRITE_SPIDER_600, self.consts.SPRITE_SPIDER_600_FRAMES, step_counter)
            frame_spider900_idx = get_frame(SPRITE_SPIDER_900, self.consts.SPRITE_SPIDER_900_FRAMES, step_counter)
            frame_flea_idx = get_frame(SPRITE_FLEA, self.consts.SPRITE_FLEA_FRAMES, step_counter)
            frame_scorpion_idx = get_frame(SPRITE_SCORPION, self.consts.SPRITE_SCORPION_FRAMES, step_counter)
            # frame_sparks_idx = get_frame(SPRITE_SPARKS, self.consts.SPRITE_SPARKS_FRAMES, step_counter)
            frame_bottom_border_idx = get_frame(SPRITE_BOTTOM_BORDER, self.consts.SPRITE_BOTTOM_BORDER_FRAMES, step_counter)

            # --- Recoloring depending on wave --- #
            recolored_sparks = get_sparks(state.death_counter)

            def wave_0():
                return (
                    recolor_sprite(frame_player_idx, self.consts.ORANGE),
                    recolor_sprite(frame_player_spell_idx, self.consts.ORANGE),
                    recolor_sprite(frame_centipede_idx, self.consts.PINK),
                    recolor_sprite(frame_mushroom_idx, self.consts.ORANGE),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.PURPLE),
                    recolor_sprite(frame_spider300_idx, self.consts.PURPLE),
                    recolor_sprite(frame_spider600_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider900_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_flea_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_scorpion_idx, self.consts.LIGHT_BLUE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.GREEN),
                )

            def wave_1():
                return (
                    recolor_sprite(frame_player_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_player_spell_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_centipede_idx, self.consts.RED),
                    recolor_sprite(frame_mushroom_idx, self.consts.DARK_BLUE),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider300_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider600_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider900_idx, self.consts.ORANGE),
                    recolor_sprite(frame_flea_idx, self.consts.YELLOW),
                    recolor_sprite(frame_scorpion_idx, self.consts.ORANGE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.LIGHT_BLUE),
                )

            def wave_2():
                return (
                    recolor_sprite(frame_player_idx, self.consts.YELLOW),
                    recolor_sprite(frame_player_spell_idx, self.consts.YELLOW),
                    recolor_sprite(frame_centipede_idx, self.consts.PURPLE),
                    recolor_sprite(frame_mushroom_idx, self.consts.YELLOW),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider300_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider600_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider900_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_flea_idx, self.consts.PINK),
                    recolor_sprite(frame_scorpion_idx, self.consts.DARK_BLUE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.ORANGE),
                )

            def wave_3():
                return (
                    recolor_sprite(frame_player_idx, self.consts.PINK),
                    recolor_sprite(frame_player_spell_idx, self.consts.PINK),
                    recolor_sprite(frame_centipede_idx, self.consts.GREEN),
                    recolor_sprite(frame_mushroom_idx, self.consts.PINK),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider300_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider600_idx, self.consts.DARK_PURPLE),
                    recolor_sprite(frame_spider900_idx, self.consts.YELLOW),
                    recolor_sprite(frame_flea_idx, self.consts.RED),
                    recolor_sprite(frame_scorpion_idx, self.consts.YELLOW),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.DARK_PURPLE),
                )

            def wave_4():
                return (
                    recolor_sprite(frame_player_idx, self.consts.RED),
                    recolor_sprite(frame_player_spell_idx, self.consts.RED),
                    recolor_sprite(frame_centipede_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_mushroom_idx, self.consts.RED),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_spider300_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_spider600_idx, self.consts.DARK_YELLOW),
                    recolor_sprite(frame_spider900_idx, self.consts.PINK),
                    recolor_sprite(frame_flea_idx, self.consts.PURPLE),
                    recolor_sprite(frame_scorpion_idx, self.consts.PINK),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.DARK_YELLOW),
                )

            def wave_5():
                return (
                    recolor_sprite(frame_player_idx, self.consts.PURPLE),
                    recolor_sprite(frame_player_spell_idx, self.consts.PURPLE),
                    recolor_sprite(frame_centipede_idx, self.consts.ORANGE),
                    recolor_sprite(frame_mushroom_idx, self.consts.PURPLE),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.YELLOW),
                    recolor_sprite(frame_spider300_idx, self.consts.YELLOW),
                    recolor_sprite(frame_spider600_idx, self.consts.PINK),
                    recolor_sprite(frame_spider900_idx, self.consts.RED),
                    recolor_sprite(frame_flea_idx, self.consts.GREEN),
                    recolor_sprite(frame_scorpion_idx, self.consts.RED),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.PINK),
                )

            def wave_6():
                return (
                    recolor_sprite(frame_player_idx, self.consts.GREEN),
                    recolor_sprite(frame_player_spell_idx, self.consts.GREEN),
                    recolor_sprite(frame_centipede_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_mushroom_idx, self.consts.GREEN),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.PINK),
                    recolor_sprite(frame_spider300_idx, self.consts.PINK),
                    recolor_sprite(frame_spider600_idx, self.consts.RED),
                    recolor_sprite(frame_spider900_idx, self.consts.PURPLE),
                    recolor_sprite(frame_flea_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_scorpion_idx, self.consts.PURPLE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.RED),
                )

            def wave_7():
                return (
                    recolor_sprite(frame_player_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_player_spell_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_centipede_idx, self.consts.YELLOW),
                    recolor_sprite(frame_mushroom_idx, self.consts.LIGHT_BLUE),
                    frame_poisoned_mushroom_idx,
                    recolor_sprite(frame_spider_idx, self.consts.RED),
                    recolor_sprite(frame_spider300_idx, self.consts.RED),
                    recolor_sprite(frame_spider600_idx, self.consts.RED),
                    recolor_sprite(frame_spider900_idx, self.consts.GREEN),
                    recolor_sprite(frame_flea_idx, self.consts.ORANGE),
                    recolor_sprite(frame_scorpion_idx, self.consts.GREEN),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.RED),
                )

            wave_mod = wave[1] % 8

            return jax.lax.switch(
                wave_mod,
                [wave_0, wave_1, wave_2, wave_3, wave_4, wave_5, wave_6, wave_7],
            )

        (
            frame_player,
            frame_player_spell,
            frame_centipede,
            frame_mushroom,
            frame_poisoned_mushroom,
            frame_spider,
            frame_spider300,
            frame_spider600,
            frame_spider900,
            frame_flea,
            frame_scorpion,
            frame_sparks,
            frame_bottom_border,
        ) = get_sprite_frames(state.wave, state.step_counter)

        ### -------- Render player -------- ###
        raster = jnp.where(
            state.death_counter >= 0,
            jru.render_at(
                raster,
                state.player_x,
                state.player_y,
                frame_player,
            ),
            raster
        )

        ### -------- Render player spell -------- ###
        raster = jnp.where(
            state.player_spell[2] != 0,
            jru.render_at(
                raster,
                state.player_spell[0],
                state.player_spell[1],
                frame_player_spell,
            ),
            raster
        )

        ### -------- Render mushrooms -------- ###
        def render_mushrooms(i, raster_base):
            alive = state.mushroom_positions[i][3] > 0
            poisoned = state.mushroom_positions[i][2] == 1

            return jax.lax.cond(
                alive,
                lambda r: jax.lax.cond(
                    poisoned,
                    lambda r2: jru.render_at(
                        r2,
                        state.mushroom_positions[i][0],
                        state.mushroom_positions[i][1],
                        frame_poisoned_mushroom,
                    ),
                    lambda r2: jru.render_at(
                        r2,
                        state.mushroom_positions[i][0],
                        state.mushroom_positions[i][1],
                        frame_mushroom,
                    ),
                    r,
                ),
                lambda r: r,
                raster_base,
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_MUSHROOMS, render_mushrooms, raster)

        ### -------- Render centipede -------- ###
        def render_centipede_segment(i, raster_base):
            should_render = jnp.logical_and(state.centipede_position[i][4] != 0, state.death_counter <= 0)
            return jax.lax.cond(
                should_render,
                lambda r: jru.render_at(
                    r,
                    state.centipede_position[i][0],
                    state.centipede_position[i][1],
                    frame_centipede,
                ),
                lambda r: r,
                raster_base
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_SEGMENTS, render_centipede_segment, raster)

        ### -------- Render spider -------- ###
        raster = jnp.where(
            state.spider_position[2] != 0,
            jru.render_at(
                raster,
                state.spider_position[0] + 2,
                state.spider_position[1] - 2,
                frame_spider,
            ),
            raster
        )

        ### -------- Render spider score -------- ###
        raster = jnp.where(
            state.spider_points[1] != 0,
            jru.render_at(
                raster,
                state.spider_position[0] + 2,
                state.spider_position[1] - 2,
                jnp.where(
                    state.spider_points[0] == 1,
                    frame_spider300,
                    jnp.where(
                        state.spider_points[0] == 2,
                        frame_spider600,
                        frame_spider900,
                    )
                )
            ),
            raster
        )

        ### -------- Render Flea -------- ###
        raster = jnp.where(
            state.flea_position[2] != 0,
            jru.render_at(
                raster,
                state.flea_position[0],
                state.flea_position[1],
                frame_flea,
            ),
            raster
        )

        ### -------- Render Scorpion -------- ###
        raster = jnp.where(
            state.scorpion_position[2] != 0,
            jru.render_at(
                raster,
                state.scorpion_position[0] + 2,
                state.scorpion_position[1] - 2,
                frame_scorpion,
                flip_horizontal=state.scorpion_position[2] == -1,
            ),
            raster
        )

        ### -------- Render sparks -------- ###
        def render_sparks(
                frame_sparks,
                raster_base,
                player_pos,
                mush_pos,
                death_counter
        ):
            def no_render():
                return raster_base

            def player_sparks():
                return jru.render_at(
                    raster_base,
                    player_pos[0] - 4,
                    player_pos[1] + 3,
                    frame_sparks,
                )

            def mushroom_sparks():
                mush_alive = jnp.count_nonzero(mush_pos[:, 3])
                alive_idx = mush_alive - jnp.ceil(death_counter / 4)

                def get_mushroom():
                    idx_y = jnp.argsort(mush_pos[:, 1])
                    mush_sorted_y = mush_pos[idx_y]
                    idx_x = jnp.argsort(mush_sorted_y[:, 0])
                    mush_sorted_x = mush_sorted_y[idx_x]

                    def body(i, carry):
                        existing_mush_idx, mushroom = carry
                        cond1 = jnp.all(mushroom == 0)  # "empty" marker for (4,) vector
                        cond2 = mush_sorted_x[i, 3] != 0
                        cond3 = existing_mush_idx == alive_idx

                        # Branch result if cond3
                        new_existing_idx = jnp.where(cond3, existing_mush_idx, existing_mush_idx + 1)
                        new_mushroom = jnp.where(cond3, mush_sorted_x[i], mushroom)

                        # Combine cond2
                        new_existing_idx = jnp.where(cond2, new_existing_idx, existing_mush_idx)
                        new_mushroom = jnp.where(cond2, new_mushroom, mushroom)

                        # Combine cond1
                        new_existing_idx = jnp.where(cond1, new_existing_idx, existing_mush_idx)
                        new_mushroom = jnp.where(cond1, new_mushroom, mushroom)

                        return new_existing_idx, new_mushroom

                    # init_carry: index scalar, and a "zero row" with same shape as mush_pos row
                    init_carry = (jnp.array(0), jnp.zeros(mush_pos.shape[1], dtype=mush_pos.dtype))
                    _, res_mushroom = jax.lax.fori_loop(0, mush_sorted_x.shape[0], body, init_carry)
                    return res_mushroom

                mush = get_mushroom()

                res_raster = jru.render_at(
                    raster_base,
                    mush[0] - 2,
                    mush[1] - 2,
                    frame_sparks,
                )

                return res_raster

            return jax.lax.cond(
                death_counter != 0,
                lambda: jax.lax.cond(
                    death_counter < 0,
                    lambda: player_sparks(),
                    lambda: jax.lax.cond(
                        (death_counter - 1) % 4 >= 2,
                        lambda: mushroom_sparks(),
                        lambda: no_render(),
                    ),
                ),
                lambda: no_render(),
            )

        raster = render_sparks(
            frame_sparks,
            raster,
            jnp.array([state.player_x, state.player_y]),
            state.mushroom_positions,
            state.death_counter,
        )


        ### -------- Render bottom border -------- ###
        raster = jru.render_at(
            raster,
            16,
            183,
            frame_bottom_border,
        )

        ### -------- Render score -------- ###
        score_array = jru.int_to_digits(state.score, max_digits=6)
        # first_nonzero = jnp.argmax(score_array != 0)
        # _, score_array = jnp.split(score_array, first_nonzero - 1)
        raster = jru.render_label(raster, 100, 187, score_array, DIGITS, spacing=8)

        ### -------- Render live indicator -------- ###
        raster = jru.render_indicator(raster, 16, 187, state.lives - 1, LIFE_INDICATOR, spacing=8)

        return raster