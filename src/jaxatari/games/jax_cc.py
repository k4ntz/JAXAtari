import os
from functools import partial
import jax
import jax.numpy as jnp
import pygame
import jaxatari.rendering.atraJaxis as aj
import chex
from typing import NamedTuple

# Game Constants
WIDTH = 160
HEIGHT = 192
SCALING_FACTOR = 3

# Colors
BACKGROUND_COLOR = (181,124,29)

# Define action space
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

# First wave directions from original code
FIRST_WAVE_DIRS = jnp.array([False, False, False, True])

class SpawnState(NamedTuple):
    difficulty: chex.Array  # Current difficulty level (0-7)
    lane_dependent_pattern: chex.Array  # Track waves independently per lane [4 lanes]
    to_be_spawned: (
        chex.Array
    )  # tracks which enemies are still in the spawning cycle [4 lanes * 3 slots] -> necessary due to the spaced out spawning of multiple enemies
    survived: (
        chex.Array
    )  # track if last enemy survived [4 lanes * 3 slots] -> 1 if survived whilst going right, 0 if not, -1 if survived whilst going left
    prev_sub: chex.Array  # Track previous entity type for each lane [4 lanes]
    spawn_timers: chex.Array  # Individual spawn timers per lane [4 lanes]
    diver_array: (
        chex.Array
    )  # Track which divers are still in the spawning cycle [4 lanes]
    lane_directions: (
        chex.Array
    )  # Track lane directions for each wave [4 lanes] -> 0 = right, 1 = left

def initialize_spawn_state() -> SpawnState:
    """Initialize spawn state with first wave matching original game."""
    return SpawnState(
        difficulty=jnp.array(0),
        lane_dependent_pattern=jnp.zeros(
            4, dtype=jnp.int32
        ),  # Each lane starts at wave 0
        to_be_spawned=jnp.zeros(
            12, dtype=jnp.int32
        ),  # Track which enemies are still in the spawning cycle
        survived=jnp.zeros(12, dtype=jnp.int32),  # Track which enemies survived
        prev_sub=jnp.ones(
            4, dtype=jnp.int32
        ),  # Track previous entity type (0 if shark, 1 if sub) -> starts at 1 since the first wave is sharks
        spawn_timers=jnp.array(
            [277, 277, 277, 277 + 60], dtype=jnp.int32
        ),  # 277 is the std starting timer in the base game
        diver_array=jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        lane_directions=FIRST_WAVE_DIRS.astype(jnp.int32),  # First wave directions
    )

# Game state container
class ChopperCommandState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array    # 0 for right, 1 for left
    score: chex.Array
    lives: chex.Array
    spawn_state: SpawnState
    enemy_chopper_positions: chex.Array  # (4, 3) array for enemy choppers
    jet_positions: (
        chex.Array
    )  # (12, 3) array for jets - separated into 4 lanes, 3 slots per lane [left to right]
    enemy_bomb_positions: (
        chex.Array
    )  # (4, 3) array for enemy missiles (only the front boats can shoot)
    surface_chopper_position: chex.Array  # (1, 3) array for surface submarine
    player_missile_position: (
        chex.Array
    )  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    successful_rescues: (
        chex.Array
    )  # Number of times the player has surfaced with all six divers
    death_counter: chex.Array  # Counter for tracking death animation
    obs_stack: chex.ArrayTree  # Observation stack for frame stacking
    rng_key: chex.PRNGKey

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load sprites - no padding needed for background since it's already full size
    bg1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/bg/1.npy"))
    pl_chopper1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/1.npy"))
    pl_chopper2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/2.npy"))

    # Pad player helicopter sprites to match each other
    pl_heli_sprites = aj.pad_to_match([pl_chopper1, pl_chopper2])

    # Background sprite (no padding needed)
    SPRITE_BG = jnp.expand_dims(bg1, axis=0)

    # Player helicopter sprites
    SPRITE_PL_HELI = jnp.concatenate(
        [
            jnp.repeat(pl_heli_sprites[0][None], 4, axis=0),
            jnp.repeat(pl_heli_sprites[1][None], 4, axis=0),
        ]
    )

    return (
        SPRITE_BG,
        SPRITE_PL_HELI,
    )

# Load sprites once at module level
(
    SPRITE_BG,
    SPRITE_PL_HELI,
) = load_sprites()

from jaxatari.renderers import AtraJaxisRenderer

class Renderer_AtraJaxis(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self): #(, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        #render background
        frame_bg = aj.get_sprite_frame(SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # render player chopper
        frame_pl_chopper = aj.get_sprite_frame(SPRITE_PL_HELI, 0)
        raster = aj.render_at(raster, 0, 0, frame_pl_chopper)

        return raster

if __name__ == "__main__":
    # Initialize game and renderer
    # game = ?
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    pygame.display.set_caption("Chopper Command")

    renderer_AtraJaxis = Renderer_AtraJaxis()

    running = True

    while running:
        screen.fill(BACKGROUND_COLOR)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #pygame.display.update()
        raster = renderer_AtraJaxis.render() #(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)

    pygame.quit()


    def get_human_action() -> chex.Array:

        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        fire = keys[pygame.K_SPACE]

        # Diagonal movements with fire
        if up and right and fire:
            return jnp.array(UPRIGHTFIRE)
        if up and left and fire:
            return jnp.array(UPLEFTFIRE)
        if down and right and fire:
            return jnp.array(DOWNRIGHTFIRE)
        if down and left and fire:
            return jnp.array(DOWNLEFTFIRE)

        # Cardinal directions with fire
        if up and fire:
            return jnp.array(UPFIRE)
        if down and fire:
            return jnp.array(DOWNFIRE)
        if left and fire:
            return jnp.array(LEFTFIRE)
        if right and fire:
            return jnp.array(RIGHTFIRE)

        # Diagonal movements
        if up and right:
            return jnp.array(UPRIGHT)
        if up and left:
            return jnp.array(UPLEFT)
        if down and right:
            return jnp.array(DOWNRIGHT)
        if down and left:
            return jnp.array(DOWNLEFT)

        # Cardinal directions
        if up:
            return jnp.array(UP)
        if down:
            return jnp.array(DOWN)
        if left:
            return jnp.array(LEFT)
        if right:
            return jnp.array(RIGHT)
        if fire:
            return jnp.array(FIRE)

        return jnp.array(NOOP)