"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""
#TODO: replace fori_loops with vmap where possible

import os
from functools import partial
from typing import Tuple, NamedTuple, Callable
import jax
import jax.numpy as jnp
import chex
import pygame
import jaxatari.rendering.atraJaxis as aj
import numpy as np
from gymnax.environments import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer

# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3
DEATH_PAUSE_FRAMES = 60

WIDTH = 160
HEIGHT = 192
SCALING_FACTOR = 4

# Chopper Constants TODO: Tweak these to match feeling of real game
ACCEL = 0.03  # DEFAULT: 0.05 | how fast the chopper accelerates
FRICTION = 0.02  # DEFAULT: 0.02 | how fast the chopper decelerates
MAX_VELOCITY = 3.0  # DEFAULT: 3.0 | maximum speed
DISTANCE_WHEN_FLYING = 10 # DEFAULT: 10 | How far the chopper moves towards the middle when flying for a longer amount of time
LOCAL_PLAYER_OFFSET_SPEED = 1 # DEFAULT: 1 | How fast the chopper changes the on-screen position when changing its facing direction
ALLOW_MOVE_OFFSET = 13 # DEFAULT: 13 | While in the no_move_pause, this is the offset measured from the left screen border where moving the chopper (exiting the pause) is allowed again
PLAYER_ROTOR_SPEED = 3 # DEFAULT: 3 | The smaller this value, the faster the rotor blades of the player chopper spin

# Score
SCORE_PER_JET_KILL = 200
SCORE_PER_CHOPPER_KILL = 100

# Player Missile Constants
PLAYER_MISSILE_WIDTH = 80 # Sprite size_x
MISSILE_COOLDOWN_FRAMES = 8  # DEFAULT: 8 | How fast Chopper can shoot (higher is slower) TODO: Das müssen wir ändern und höher machen bei dem schweren Schwierigkeitsgrad
MISSILE_SPEED = 10 # DEFAULT: 10 | Missile speed (higher is faster)#TODO: tweak MISSILE_SPEED and MISSILE_COOLDOWN_FRAMES to match real game (already almost perfect)
MISSILE_ANIMATION_SPEED = 6 # DEFAULT: 6 | Rate at which missile changes sprite textures (based on traveled distance of missile)

# Enemy Missile Constants
ENEMY_MISSILE_SPAWN_PROBABILITY = 0.01 # The probability that an enemy missiles spawns at one of the living enemies in each frame, if the missile of the giving enemy is not "alive". (Meaning that for 0.01 for example, an enemy shoots a missile on average 100 frames after its previous missile died).
ENEMY_MISSILE_SPLIT_PROBABILITY = 0.01 # The probability that an enemy missiles splits in a frame
ENEMY_MISSILE_MAXIMUM_Y_SPEED_BEFORE_SPLIT = 0.75 # Maximum speed (+ and -) of a missile before split. This means that for 2 for example, the missiles will have speeds between -2 and 2 (chosen randomly).
ENEMY_MISSILE_Y_SPEED_AFTER_SPLIT = 2.5 #TODO: Make match real game

# Colors
BACKGROUND_COLOR = (0, 0, 139)  # Dark blue for sky
PLAYER_COLOR = (187, 187, 53)  # Yellow for player helicopter
ENEMY_COLOR = (170, 170, 170)  # Gray for enemy helicopters
MISSILE_COLOR = (255, 255, 255)  # White for missiles
SCORE_COLOR = (210, 210, 64)  # Score color

# Object sizes and initial positions
PLAYER_SIZE = (16, 9)  # Width, Height
TRUCK_SIZE = (8, 7)
JET_SIZE = (8, 6)
CHOPPER_SIZE = (8, 9)
PLAYER_MISSILE_SIZE = (80, 1) #Default (80, 1)
ENEMY_MISSILE_SIZE = (2, 1)

PLAYER_START_X = 0
PLAYER_START_Y = 100

X_BORDERS = (0, 160)
PLAYER_BOUNDS = (0, 160), (52, 150)

# Maximum number of objects
MAX_TRUCKS = 12
MAX_JETS = 12
MAX_CHOPPERS = 12
MAX_ENEMIES = 12
MAX_PLAYER_MISSILES = 2 #TODO: I think the real game uses one player missile max on screen. This feels nicer though
MAX_ENEMY_MISSILES = MAX_ENEMIES * 2 * 2 # Two missiles for every enemy (jets and choppers) (this does not mean, that there are always this many missiles on the screen/in the game)

# Enemies movement
JET_VELOCITY_LEFT = 1.5
JET_VELOCITY_RIGHT = 1
CHOPPER_VELOCITY_LEFT = 0.8
CHOPPER_VELOCITY_RIGHT = 0.5
OUT_OF_CYCLE_RIGHT = 64
OUT_OF_CYCLE_LEFT = 128

# Minimap
MINIMAP_WIDTH = 48
MINIMAP_HEIGHT = 16

MINIMAP_POSITION_X = (WIDTH // 2) - (MINIMAP_WIDTH // 2) #TODO: Im echten Game wird die Minimap nicht mittig, sondern weiter links gerendert. Wir müssen besprechen ob wir das auch machen, dann müsste man nur diese Zahl hier ändern (finde es aber so schöner)
MINIMAP_POSITION_Y = 165

DOWNSCALING_FACTOR_WIDTH = WIDTH // MINIMAP_WIDTH
DOWNSCALING_FACTOR_HEIGHT = HEIGHT // MINIMAP_HEIGHT

#Object rendering
TRUCK_SPAWN_DISTANCE = 248 # distance 240px + truck width

FRAMES_DEATH_ANIMATION_ENEMY = 16
FRAMES_DEATH_ANIMATION_TRUCK = 32 #TODO: Make match real game
TRUCK_FLICKER_RATE = 3 #TODO: Make match real game

PLAYER_FADE_OUT_START_THRESHOLD_0 = 0.25
PLAYER_FADE_OUT_START_THRESHOLD_1 = 0.125

# define object orientations
FACE_LEFT = -1
FACE_RIGHT = 1

SPAWN_POSITIONS_Y = jnp.array([60, 90, 120])
TRUCK_SPAWN_POSITIONS = 156


class ChopperCommandState(NamedTuple):
    player_x: chex.Array                 # x-coordinate of the player’s chopper in world space
    player_y: chex.Array                 # y-coordinate of the player’s chopper in world space
    player_velocity_x: chex.Array        # horizontal velocity (momentum) of the player’s chopper; positive = moving right, negative = moving left
    local_player_offset: chex.Array      # offset of the player’s chopper from the screen center (used for scrolling logic and chopper’s on-screen position)
    player_facing_direction: chex.Array  # current facing direction of the player’s chopper: -1 = facing left, +1 = facing right, 0 = invalid
    score: chex.Array                    # current game score/points
    lives: chex.Array                    # number of remaining lives for the player
    save_lives: chex.Array               # keeps track of how often the player was granted a life for reaching another 10000 score points
    truck_positions: chex.Array          # shape (MAX_TRUCKS, 4): for each truck, stores [x, y, direction (active flag), death_timer]
    jet_positions: chex.Array            # shape (MAX_JETS, 4): for each enemy jet, stores [x, y, direction (active flag), death_timer]
    chopper_positions: chex.Array        # shape (MAX_ENEMIES, 4): for each enemy chopper, stores [x, y, direction (active flag), death_timer]
    enemy_missile_positions: chex.Array  # shape (MAX_MISSILES, 4): for each enemy missile, stores [x, y, direction, did_split_flag]
    player_missile_positions: chex.Array # shape (MAX_MISSILES, 4): for each player missile, stores [x, y, direction, x_coordinate of spawn point]
    player_missile_cooldown: chex.Array  # cooldown timer until the player can fire the next missile
    player_collision: chex.Array         # boolean flag indicating whether the player has collided this frame
    step_counter: chex.Array             # total number of game ticks/frames elapsed so far
    pause_timer: chex.Array              # counter for how many frames remain in the game pause before respawning; 0 = fully dead, respawn initiated, 1 = either no lives left, infinite pause or ->, 1 - DEATH_PAUSE_FRAMES: counting down for the duration of pause, DEATH_PAUSE_FRAMES + 1 = death_pause, DEATH_PAUSE_FRAMES + 2 = no_move_pause
    obs_stack: chex.ArrayTree            # stacked sequence of past observations (for frame‐stacking in the agent)
    rng_key: chex.PRNGKey                # current PRNG key for any stochastic operations (e.g., random enemy spawns)

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class ChopperCommandObservation(NamedTuple):
    player: EntityPosition
    trucks: jnp.ndarray # Shape (MAX_TRUCKS, 5) - MAX_TRUCKS enemies, each with x,y,w,h,active
    jets: jnp.ndarray  # Shape (MAX_JETS, 5) - MAX_JETS enemies, each with x,y,w,h,active
    choppers: jnp.ndarray # Shape (MAX_CHOPPERS, 5) - MAX_CHOPPERS enemies, each with x,y,w,h,active
    enemy_missiles: jnp.ndarray  # Shape (MAX_MISSILES, 5)
    player_missile: EntityPosition
    player_score: jnp.ndarray
    lives: jnp.ndarray

class ChopperCommandInfo(NamedTuple):
    step_counter: jnp.ndarray  # Current step count
    all_rewards: jnp.ndarray  # All rewards for the current step

# RENDER CONSTANTS
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load sprites - no padding needed for background since it's already full size
    pl_chopper1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/1.npy"))
    pl_chopper2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/2.npy"))
    friendly_truck1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/friendly_truck/1.npy"))
    friendly_truck2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/friendly_truck/2.npy"))
    enemy_jet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/enemy_jet/normal.npy"))
    enemy_chopper1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/enemy_chopper/1.npy"))
    enemy_chopper2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/enemy_chopper/2.npy"))
    enemy_bomb = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/bomb/1.npy"))

    bg_sprites = []
    for i in range(1, 161):
        temp = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/choppercommand/bg/{i}.npy"))
        bg_sprites.append(temp)
        bg_sprites[i - 1] = jnp.expand_dims(bg_sprites[i - 1], axis=0)

    pl_missile_sprites_temp = []
    for i in range(0, 16):
        temp = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/choppercommand/player_missiles/missile_{i}.npy"))
        pl_missile_sprites_temp.append(temp)
        pl_missile_sprites_temp[i] = jnp.expand_dims(pl_missile_sprites_temp[i], axis=0)

    minimap_mountains_temp = []
    for i in range(1, 9):
        temp = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/choppercommand/minimap/mountains/{i}.npy"))
        minimap_mountains_temp.append(temp)
        minimap_mountains_temp[i - 1] = jnp.expand_dims(minimap_mountains_temp[i - 1], axis=0)

    # Pad player helicopter sprites to match each other
    pl_chopper_sprites = aj.pad_to_match([pl_chopper1, pl_chopper2])

    # Pad friendly truck sprites to match each other
    friendly_truck_sprites = aj.pad_to_match([friendly_truck1, friendly_truck2])

    # Pad enemy jet sprites to match each other
    enemy_jet_sprites = [enemy_jet]

    # Pad enemy helicopter sprites to match each other
    enemy_heli_sprites = aj.pad_to_match([enemy_chopper1, enemy_chopper2])

    # Pad player missile sprites to match each other
    pl_missile_sprites = pl_missile_sprites_temp

    # Pad enemy missile sprites to match each other
    enemy_missile_sprites = [enemy_bomb]

    # Background sprite (no padding needed)
    SPRITE_BG = jnp.concatenate(bg_sprites, axis=0) # jnp.expand_dims(bg1, axis=0)

    # Player helicopter sprites
    SPRITE_PL_CHOPPER = jnp.concatenate(
        [
            jnp.repeat(pl_chopper_sprites[0][None], PLAYER_ROTOR_SPEED, axis=0),
            jnp.repeat(pl_chopper_sprites[1][None], PLAYER_ROTOR_SPEED, axis=0),
        ]
    )

    # Friendly truck sprites
    SPRITE_FRIENDLY_TRUCK = jnp.concatenate(
        [
            jnp.repeat(friendly_truck_sprites[0][None], 4, axis=0),
            jnp.repeat(friendly_truck_sprites[1][None], 4, axis=0),
        ]
    )

    # Enemy jet sprite
    SPRITE_ENEMY_JET = jnp.repeat(enemy_jet_sprites[0][None], 1, axis=0)

    # Enemy helicopter sprites
    SPRITE_ENEMY_CHOPPER = jnp.concatenate(
        [
            jnp.repeat(enemy_heli_sprites[0][None], 4, axis=0),
            jnp.repeat(enemy_heli_sprites[1][None], 4, axis=0),
        ]
    )

    DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/choppercommand/score/{}.npy"))
    LIFE_INDICATOR = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/score/chopper.npy"))

    #Death Sprites
    PLAYER_DEATH_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/death_1.npy"))
    PLAYER_DEATH_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/death_2.npy"))
    PLAYER_DEATH_3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/player_chopper/death_3.npy"))

    ENEMY_DEATH_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/enemy_death/death_1.npy"))
    ENEMY_DEATH_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/enemy_death/death_2.npy"))
    ENEMY_DEATH_3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/enemy_death/death_3.npy"))

    # Player missile sprites
    SPRITE_PL_MISSILE = jnp.concatenate(pl_missile_sprites, axis=0)

    # Enemy missile sprites
    SPRITE_ENEMY_MISSILE = jnp.repeat(enemy_missile_sprites[0][None], 1, axis=0)

    #Everything having to do with the minimap
    MINIMAP_BG = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/minimap/background.npy"))
    MINIMAP_PLAYER = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/minimap/player.npy"))
    MINIMAP_TRUCK = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/minimap/truck.npy"))
    MINIMAP_ACTIVISION_LOGO = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/minimap/activision_logo.npy")) #delete if necessary
    MINIMAP_ENEMY = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/choppercommand/minimap/enemy.npy"))
    MINIMAP_MOUNTAINS = jnp.concatenate(minimap_mountains_temp, axis=0)


    return (
        SPRITE_BG,
        SPRITE_PL_CHOPPER,
        SPRITE_FRIENDLY_TRUCK,
        SPRITE_ENEMY_JET,
        SPRITE_ENEMY_CHOPPER,
        SPRITE_PL_MISSILE,
        SPRITE_ENEMY_MISSILE,
        DIGITS,
        LIFE_INDICATOR,
        PLAYER_DEATH_1,
        PLAYER_DEATH_2,
        PLAYER_DEATH_3,
        ENEMY_DEATH_1,
        ENEMY_DEATH_2,
        ENEMY_DEATH_3,
        MINIMAP_BG,
        MINIMAP_MOUNTAINS,
        MINIMAP_PLAYER,
        MINIMAP_TRUCK,
        MINIMAP_ENEMY,
        MINIMAP_ACTIVISION_LOGO, #delete if necessary
    )

# Load sprites once at module level
(
    SPRITE_BG,
    SPRITE_PL_CHOPPER,
    SPRITE_FRIENDLY_TRUCK,
    SPRITE_ENEMY_JET,
    SPRITE_ENEMY_HELI,
    SPRITE_PL_MISSILE,
    SPRITE_ENEMY_MISSILE,
    DIGITS,
    LIFE_INDICATOR,
    PLAYER_DEATH_1,
    PLAYER_DEATH_2,
    PLAYER_DEATH_3,
    ENEMY_CHOPPER_DEATH_1,
    ENEMY_CHOPPER_DEATH_2,
    ENEMY_CHOPPER_DEATH_3,
    MINIMAP_BG,
    MINIMAP_MOUNTAINS,
    MINIMAP_PLAYER,
    MINIMAP_TRUCK,
    MINIMAP_ENEMY,
    MINIMAP_ACTIVISION_LOGO, #delete if necessary
) = load_sprites()

@jax.jit
def check_collision_single(pos1, size1, pos2, size2):
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

@jax.jit
def check_collision_batch(pos1, size1, pos2_array, size2):
    """Check collision between one entity and an array of entities"""
    # Calculate edges for rectangle 1
    rect1_left = pos1[0]
    rect1_right = pos1[0] + size1[0]
    rect1_top = pos1[1]
    rect1_bottom = pos1[1] + size1[1]

    # Calculate edges for all rectangles in pos2_array
    rect2_left = pos2_array[:, 0]
    rect2_right = pos2_array[:, 0] + size2[0]
    rect2_top = pos2_array[:, 1]
    rect2_bottom = pos2_array[:, 1] + size2[1]

    # Check overlap for all entities
    horizontal_overlaps = jnp.logical_and(
        rect1_left < rect2_right,
        rect1_right > rect2_left
    )

    vertical_overlaps = jnp.logical_and(
        rect1_top < rect2_bottom,
        rect1_bottom > rect2_top
    )

    # Combine checks for each entity
    collisions = jnp.logical_and(horizontal_overlaps, vertical_overlaps)

    # Return true if any collision detected
    return jnp.any(collisions)

def kill_entity(
        enemy_pos: chex.Array,
        death_timer: int
        ) -> chex.Array:
    return jnp.array([
        enemy_pos[0],  # x
        enemy_pos[1],  # y
        enemy_pos[2],  # direction
        death_timer  # death_timer
    ], dtype=enemy_pos.dtype)

@jax.jit
def check_missile_collisions(
    missile_positions: chex.Array,  # (MAX_MISSILES, 4)
    enemy_positions: chex.Array,    # (N_ENEMIES, 2)
    score: chex.Array,
    rng_key: chex.PRNGKey,
    on_screen_position: chex.Array,
    player_x: chex.Array,
    enemy_size: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
    """Check for collisions between player missiles and enemies, mit dynamischer Breitenanpassung."""

    def check_single_missile(missile_idx, carry):
        missile_positions, enemy_positions, score, rng_key = carry

        missile = missile_positions[missile_idx]
        missile_x, missile_y, direction, _ = missile
        missile_active = missile[3] != 0

        def check_single_enemy(enemy_idx, inner_carry):
            missile_positions, enemy_positions, score, rng_key = inner_carry
            enemy_pos = enemy_positions[enemy_idx]
            enemy_active = enemy_pos[3] > FRAMES_DEATH_ANIMATION_ENEMY

            # Sichtfeldgrenzen
            left_bound = player_x - on_screen_position
            right_bound = left_bound + WIDTH

            missile_left = missile_x
            missile_right = missile_x + PLAYER_MISSILE_SIZE[0]

            # Dynamische Breite berechnen
            clipped_left = jnp.maximum(missile_left, left_bound)
            clipped_right = jnp.minimum(missile_right, right_bound)

            # Neue Breite
            clipped_width = jnp.maximum(0, clipped_right - clipped_left)

            # Missile ist zu klein → keine Kollision möglich
            too_small = clipped_width <= 0

            # Neue Position für Kollisionstest
            adjusted_pos = jnp.array([clipped_left, missile_y])
            adjusted_size = jnp.array([clipped_width, PLAYER_MISSILE_SIZE[1]])

            collision = jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(
                        missile_active,
                        enemy_active
                    ),
                    jnp.logical_not(too_small)
                ),
                check_collision_single(adjusted_pos, adjusted_size, enemy_pos, enemy_size)
            )

            #Kill initialisieren (nicht endgültig tot)
            new_enemy_pos = jnp.where(collision, kill_entity(enemy_pos, FRAMES_DEATH_ANIMATION_ENEMY), enemy_pos)

            # Punkte vergeben
            is_jet = enemy_size[1] == 6
            score_add = jnp.where(
                jnp.logical_and(collision, enemy_pos[3] > FRAMES_DEATH_ANIMATION_ENEMY),
                jnp.where(
                    is_jet,
                    SCORE_PER_JET_KILL,
                    SCORE_PER_CHOPPER_KILL
                ),
                0
            )

            # Missile deaktivieren bei Treffer
            new_missile = jnp.where(
                collision,
                jnp.array([0, 0, 0, 0], dtype=missile.dtype),
                missile
            )

            # Apply updates
            updated_enemies = enemy_positions.at[enemy_idx].set(new_enemy_pos)
            updated_missiles = missile_positions.at[missile_idx].set(new_missile)

            return (
                updated_missiles,
                updated_enemies,
                score + score_add,
                rng_key,
            )

        # Schleife über alle Gegner für eine Missile
        return jax.lax.fori_loop(
            0,
            enemy_positions.shape[0],
            check_single_enemy,
            (missile_positions, enemy_positions, score, rng_key),
        )

    # Schleife über alle Missiles
    return jax.lax.fori_loop(
        0,
        missile_positions.shape[0],
        check_single_missile,
        (missile_positions, enemy_positions, score, rng_key),
    )



@jax.jit
def check_player_collision_entity(
    player_x: chex.Array,
    player_y: chex.Array,
    player_velocity: chex.Array,
    entity_list: chex.Array,
    entity_size: Tuple[int, int],
    death_threshold: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:

    player_pos = jnp.array([player_x, player_y])
    offset = (PLAYER_SIZE[0] // 2 - entity_size[0] // 2) - (player_velocity * DISTANCE_WHEN_FLYING)

    def check_single(i, carry):
        any_collision_inner, updated_entities = carry

        world_x, world_y = entity_list[i, 0], entity_list[i, 1]
        death_timer = entity_list[i, 3]
        is_active = death_timer > death_threshold

        # Passe die Position an, wie sie auch im Renderer korrigiert wird
        adjusted_entity_pos = jnp.array([world_x + offset, world_y])

        # Prüfe Kollision nur bei aktiven Gegnern
        collision = jnp.logical_and(
            is_active,
            check_collision_single(player_pos, PLAYER_SIZE, adjusted_entity_pos, entity_size)
        )

        # Markiere getroffenen Gegner
        new_entity = jnp.where(collision, kill_entity(entity_list[i], death_threshold), entity_list[i])
        updated_entities = updated_entities.at[i].set(new_entity)

        any_collision_inner = jnp.logical_or(any_collision_inner, collision)
        return any_collision_inner, updated_entities

    initial_carry = (False, entity_list)
    any_collision, updated_entity_list = jax.lax.fori_loop(
        0,
        entity_list.shape[0],
        check_single,
        initial_carry
    )

    return any_collision, updated_entity_list


@jax.jit
def check_missile_truck_collisions(
    truck_positions: chex.Array,        # (MAX_TRUCKS, 4): [x, y, direction, death_timer]
    missile_positions: chex.Array,      # (MAX_MISSILES, 4): [x, y, direction, did_split_flag]
    truck_size: Tuple[int, int],        # (width, height)
    missile_size: Tuple[int, int],      # (width, height)
) -> Tuple[chex.Array, chex.Array]:

    def collide_one_missile(mi, carry):
        trucks, missiles = carry
        m = missiles[mi]

        # Position und aktiver Status der Missile
        mx, my, _, _ = m
        missile_active = my > 2

        # inner loop: prüfen gegen alle Trucks
        def check_truck(ti, inner):
            collided, trucks_inner = inner
            t = trucks_inner[ti]
            tx, ty, tdir, tdeath = t

            # Truck nur prüfen wenn aktiv
            truck_active = jnp.logical_and(tdir != 0, tdeath > 0)

            # Kollisionstest
            collision = jnp.logical_and(
                missile_active,
                truck_active
            ) & check_collision_single(
                jnp.array([mx, my]), missile_size,
                jnp.array([tx, ty]), truck_size
            )

            # Wenn Kollision, setze Truck death_timer auf "tot initialisieren"
            new_truck = jnp.where(
                collision,
                jnp.array([tx, ty, tdir, FRAMES_DEATH_ANIMATION_TRUCK], dtype=trucks_inner.dtype),
                t
            )
            trucks_inner = trucks_inner.at[ti].set(new_truck)
            collided = jnp.logical_or(collided, collision)
            return collided, trucks_inner

        # führe inner loop aus
        init_inner = (False, trucks)
        collided, updated_trucks = jax.lax.fori_loop(
            0, truck_positions.shape[0], check_truck, init_inner
        )

        # wenn collision: kill Missile
        new_m = jnp.where(
            collided,
            jnp.array([0.0, 0.0, 0.0, 187.0]),
            m
        )
        missiles = missiles.at[mi].set(new_m)

        return updated_trucks, missiles

    # outer loop: alle Missiles
    init = (truck_positions, missile_positions)
    updated_trucks, updated_missiles = jax.lax.fori_loop(
        0, missile_positions.shape[0], collide_one_missile, init
    )

    return updated_trucks, updated_missiles

@jax.jit
def is_slot_empty(pos: chex.Array) -> chex.Array:
    """Check if a position slot is empty (0,0,0)"""
    return pos[2] == 0

"""now can spawn enemies into fleets with different directions"""
@jax.jit
def initialize_enemy_positions(rng: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
    jet_positions = jnp.zeros((MAX_ENEMIES, 4))
    chopper_positions = jnp.zeros((MAX_ENEMIES, 4))

    fleet_start_x = -780
    fleet_spacing_x = 312
    fleet_count = 4
    units_per_fleet = 3
    vertical_spacing = 30
    y_start = HEIGHT // 2 - (units_per_fleet // 2) * vertical_spacing

    carry = (jet_positions, chopper_positions, 0, rng)

    def spawn_fleet(fleet_idx, carry):
        jet_positions, chopper_positions, global_idx, rng = carry
        rng, fleet_rng = jax.random.split(rng)

        anchor_x = fleet_start_x + fleet_idx * fleet_spacing_x

        fleet_rng, offset_rng1, offset_rng2, offset_rng3, direction_rng, chopper_rng = jax.random.split(fleet_rng, 6)

        # Für jede Einheit eine zufällige Richtung
        directions = jax.random.choice(direction_rng, jnp.array([-1, 1]), shape=(units_per_fleet,), replace=True)

        # Zufällige Anzahl Chopper
        chopper_count = jax.random.randint(chopper_rng, (), 0, units_per_fleet + 1)

        # X-Offsets wie bisher
        #offset_choices = jnp.array([-32, 0, 32])
        x_offset_array = jnp.array(
            [jax.random.randint(offset_rng1, (), -32, 32),
            jax.random.randint(offset_rng2, (), -32, 32),
            jax.random.randint(offset_rng3, (), -32, 32)])
        #x_offsets = jax.random.choice(offset_rng, offset, shape=(units_per_fleet,), replace=True)

        def place_unit(i, unit_carry):
            jet_positions, chopper_positions, jet_idx, chopper_idx = unit_carry
            y = y_start + i * vertical_spacing
            offset_x = x_offset_array[i]
            #offset_x = offset_x
            direction = directions[i]
            pos = jnp.array([anchor_x + offset_x, y, direction, FRAMES_DEATH_ANIMATION_ENEMY + 1])

            is_chopper = i < chopper_count
            chopper_positions = jax.lax.cond(
                is_chopper,
                lambda cp: cp.at[chopper_idx].set(pos),
                lambda cp: cp,
                chopper_positions
            )
            jet_positions = jax.lax.cond(
                is_chopper,
                lambda jp: jp,
                lambda jp: jp.at[jet_idx].set(pos),
                jet_positions
            )

            jet_idx += jnp.where(is_chopper, 0, 1)
            chopper_idx += jnp.where(is_chopper, 1, 0)
            return jet_positions, chopper_positions, jet_idx, chopper_idx

        jet_positions, chopper_positions, jet_idx, chopper_idx = jax.lax.fori_loop(
            0, units_per_fleet, place_unit, (jet_positions, chopper_positions, global_idx, global_idx)
        )

        new_global_idx = global_idx + units_per_fleet
        return (jet_positions, chopper_positions, new_global_idx, rng)

    jet_positions, chopper_positions, _, _ = jax.lax.fori_loop(
        0, fleet_count, spawn_fleet, carry
    )

    return jet_positions, chopper_positions

#TODO: implement just one fleet moving
@jax.jit
def step_enemy_movement(
        truck_positions: chex.Array,
        jet_positions: chex.Array,
        chopper_positions: chex.Array,
        step_counter: chex.Array,
        rng: chex.PRNGKey,
        state_player_x: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:
    rng, direction_rng = jax.random.split(rng)

    def move_enemy(pos: chex.Array, is_jet, is_in_range) -> chex.Array:
        is_active = pos[2] != 0

        def is_out_of_cycle(enemy_pos: chex.Array) -> chex.Array:
            # Auswahl der X-Positionen der mittleren trucks
            middle_trucks = jnp.array(
                [
                    truck_positions[1][0],
                    truck_positions[4][0],
                    truck_positions[7][0],
                    truck_positions[10][0],
                ]
            )

            # Berechne den absoluten Abstand zur enemy_position[0]
            distances = jnp.abs(middle_trucks - enemy_pos[0])

            # Finde den Index des kleinsten Abstands
            min_index = jnp.argmin(distances)

            nearest_middle_truck = middle_trucks[min_index]

            return jnp.logical_or(enemy_pos[0] > nearest_middle_truck + OUT_OF_CYCLE_RIGHT,
                                  enemy_pos[0] < nearest_middle_truck - OUT_OF_CYCLE_LEFT)


        def is_jet_function():

            def out_of_cycle_jet_function():

                def out_of_cycle_jet_yes_function():

                    return jax.lax.cond( # out_of_cycle_jet_yes_function
                        pos[2] == -1,
                        lambda _: jnp.array([pos[0] + pos[2] * -1 * JET_VELOCITY_LEFT - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32),
                        lambda _: jnp.array([pos[0] + pos[2] * -1 * JET_VELOCITY_RIGHT - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32),
                        operand=None
                    )

                def out_of_cycle_jet_no_function():

                    return jax.lax.cond( # out_of_cycle_jet_no_function
                        pos[2] == -1,
                        lambda _: jnp.array([pos[0] + pos[2] * JET_VELOCITY_LEFT, pos[1], pos[2], pos[3]], dtype=jnp.float32),
                        lambda _: jnp.array([pos[0] + pos[2] * JET_VELOCITY_RIGHT, pos[1], pos[2], pos[3]], dtype=jnp.float32),
                        operand=None
                    )


                return jax.lax.cond( # is_jet_function
                    is_out_of_cycle(pos),
                    lambda _: out_of_cycle_jet_yes_function(),
                    lambda _: out_of_cycle_jet_no_function(),
                    operand=None
                )


            def out_of_cycle_chopper_function():

                def out_of_cycle_chopper_yes_function():

                    return jax.lax.cond( # out_of_cycle_chopper_yes_function
                        pos[2] == -1,
                        lambda _: jnp.array([pos[0] + pos[2] * -1 * CHOPPER_VELOCITY_LEFT - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32),
                        lambda _: jnp.array([pos[0] + pos[2] * -1 * CHOPPER_VELOCITY_RIGHT - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32),
                        operand=None
                    )

                def out_of_cycle_chopper_no_function():

                    return jax.lax.cond( # out_of_cycle_chopper_no_function
                        pos[2] == -1,
                        lambda _: jnp.array([pos[0] + pos[2] * CHOPPER_VELOCITY_LEFT, pos[1], pos[2], pos[3]], dtype=jnp.float32),
                        lambda _: jnp.array([pos[0] + pos[2] * CHOPPER_VELOCITY_RIGHT, pos[1], pos[2], pos[3]], dtype=jnp.float32),
                        operand=None
                    )


                return jax.lax.cond( # out_of_cycle_chopper_function
                    is_out_of_cycle(pos),
                    lambda _: out_of_cycle_chopper_yes_function(),
                    lambda _: out_of_cycle_chopper_no_function(),
                    operand=None
                )


            return jax.lax.cond( # is_jet_function
                is_jet,
                lambda _: out_of_cycle_jet_function(),
                lambda _: out_of_cycle_chopper_function(),
                operand=None
            )


        new_pos = jax.lax.cond(
            is_active,
            lambda _: is_jet_function(),
            lambda _: pos,
            operand=None
        )

        #new_pos = jnp.where(is_active, jnp.array([new_x, pos[1], pos[2], pos[3]]), pos)

        #pos[0] + pos[2] * JET_VELOCITY

        out_of_bounds = jnp.abs(state_player_x - pos[0]) > 624
        wrapped_x = pos[0] + jnp.clip(state_player_x - pos[0], -1, 1) * 1248  # + pos[2] * 0.5
        wrapped_pos = jnp.array([wrapped_x, pos[1], pos[2], pos[3]])

        return jnp.where(out_of_bounds, wrapped_pos, new_pos)


    def is_in_range_checker(pos: chex.Array) -> chex.Array:
        return jnp.array(0)


    def process_jet(i, new_positions):
        current_pos = jet_positions[i]
        new_pos = move_enemy(current_pos, jnp.array(True), is_in_range_checker(current_pos))
        return new_positions.at[i].set(new_pos)

    new_jet_positions = jnp.zeros_like(jet_positions)
    new_jet_positions = jax.lax.fori_loop(0, jet_positions.shape[0], process_jet, new_jet_positions)

    def process_chopper(i, new_positions):
        current_pos = chopper_positions[i]
        new_pos = move_enemy(current_pos, jnp.array(False), is_in_range_checker(current_pos))
        return new_positions.at[i].set(new_pos)

    new_chopper_positions = jnp.zeros_like(chopper_positions)
    new_chopper_positions = jax.lax.fori_loop(0, chopper_positions.shape[0], process_chopper, new_chopper_positions)

    return new_jet_positions, new_chopper_positions, rng


def update_entity_death(entity_array, death_timer, is_truck):
    def update_entity(i, carry):
        entities = carry
        entity = entities[i]
        direction, timer = entity[2], entity[3]

        #Wenn tot initialisiert (also noch aktiv) und timer > 0 & timer <= FRAMES_DEATH_ANIMATION, dann dekrementieren
        new_timer = jnp.where(jnp.logical_and(timer > 0, timer <= death_timer), timer - 1, timer)

        #Nach Ablauf von death_timer Enemy deaktivieren/entfernen
        new_entity = jnp.where(
            new_timer == 0,
                    jnp.where(
                        is_truck,
                        jnp.array([entity[0], entity[1], 0, 0], dtype=entity.dtype),
                        jnp.array([0, 0, 0, 0], dtype=entity.dtype)
                    ),
            entity.at[3].set(new_timer)
        )

        return entities.at[i].set(new_entity)

    return jax.lax.fori_loop(0, entity_array.shape[0], update_entity, entity_array)


@jax.jit
def initialize_truck_positions() -> chex.Array:
    initial_truck_positions = jnp.zeros((MAX_TRUCKS, 4))
    anchor = -748
    carry = (initial_truck_positions, anchor)

    def spawn_trucks(i, carry):
        truck_positions, anchor = carry

        anchor = jnp.where(
            i % 3 == 0,
            anchor + 248,
            anchor + 32,
        )
        truck_positions = truck_positions.at[i].set(jnp.array([anchor, 156, -1, FRAMES_DEATH_ANIMATION_TRUCK + 1]))
        return truck_positions, anchor

    return jax.lax.fori_loop(0, 12, spawn_trucks, carry)[0]

@jax.jit
def step_truck_movement(
        truck_positions: chex.Array,
        state_player_x: chex.Array,
) -> chex.Array:

    # TODO: simplify if possible
    def move_single_truck(truck_pos):
        movement_x = -1 * 0.5  # Geschwindigkeit 0.5 pro Frame, egal ob tot oder nicht, weil wir die Position noch für die enemy positions brauchen

        out_of_bounds = jnp.abs(state_player_x - truck_pos[0]) > 624

        new_x = jnp.where(
            out_of_bounds,
            truck_pos[0] + jnp.sign(state_player_x - truck_pos[0]) * 1248 + movement_x,
            truck_pos[0] + movement_x,
        )

        new_pos = jnp.array([new_x, truck_pos[1], truck_pos[2], truck_pos[3]])

        return new_pos

    return jax.vmap(move_single_truck)(truck_positions)


@jax.jit
def enemy_missiles_step(
        jet_positions: chex.Array,  # (MAX_ENEMIES, 4)
        chopper_positions: chex.Array,  # (MAX_ENEMIES, 4)
        missile_states: chex.Array,  # (MAX_ENEMY_MISSILES, 4): [x, y, y_dir, did_split]
        rng: chex.PRNGKey,
) -> chex.Array:

    enemies = jnp.concat([jet_positions, chopper_positions], axis=0)

    def step_both_missiles(i, carry):
        missiles, key, eni = carry
        # Split key into spawn, split, speed, and next key
        key, key_spawn, key_split, key_speed = jax.random.split(key, 4)

        # Get current enemy
        current_enemy = enemies[eni]

        # Upper and lower missile
        missile_upper = missiles[i]
        missile_lower = missiles[i + 1]

        def maybe_spawn():
            def do_spawn():
                # Coordinates of upper missile part
                x_upper_spawn = current_enemy[0]
                y_upper_spawn = current_enemy[1]

                # Coordinates of lower missile part
                x_lower_spawn = current_enemy[0]
                y_lower_spawn = current_enemy[1] + 1

                # Random Y Velocity
                y_speed_spawn = jax.random.uniform(
                    key_speed,
                    (),
                    minval=-ENEMY_MISSILE_MAXIMUM_Y_SPEED_BEFORE_SPLIT,
                    maxval=ENEMY_MISSILE_MAXIMUM_Y_SPEED_BEFORE_SPLIT
                )

                # We also set the did_split flag to false
                spawned_upper_missile = jnp.array([x_upper_spawn, y_upper_spawn, y_speed_spawn, 187.0], dtype=jnp.float32)
                spawned_lower_missile = jnp.array([x_lower_spawn, y_lower_spawn, y_speed_spawn, 187.0], dtype=jnp.float32)

                return spawned_upper_missile, spawned_lower_missile

            return jax.lax.cond(
                jax.random.bernoulli(key_spawn, p=ENEMY_MISSILE_SPAWN_PROBABILITY),
                lambda _: do_spawn(), # Actually spawn
                lambda _: (           # Leave dead
                    jnp.array([0.0, 0.0, 0.0, 187.0], dtype=jnp.float32),
                    jnp.array([0.0, 0.0, 0.0, 187.0], dtype=jnp.float32)
                ),
                operand=None
            )

        def do_step():
            x_upper_step, y_upper_step, y_upper_speed, did_split_upper_step = missile_upper
            x_lower_step, y_lower_step, y_lower_speed, did_split_lower_step = missile_lower

            def dont_split():
                y_upper_step_inner = y_upper_step + y_upper_speed
                y_lower_step_inner = y_lower_step + y_lower_speed
                return (
                    y_upper_step_inner.astype(jnp.float32),
                    y_upper_speed.astype(jnp.float32),
                    y_lower_step_inner.astype(jnp.float32),
                    y_lower_speed.astype(jnp.float32),
                    187.0
                )

            def do_split():
                y_upper_speed_inner = jnp.array(ENEMY_MISSILE_Y_SPEED_AFTER_SPLIT, dtype=jnp.float32)
                y_lower_speed_inner = jnp.array(-ENEMY_MISSILE_Y_SPEED_AFTER_SPLIT, dtype=jnp.float32)
                y_upper_step_inner = y_upper_step + y_upper_speed_inner
                y_lower_step_inner = y_lower_step + y_lower_speed_inner
                return (
                    y_upper_step_inner.astype(jnp.float32),
                    y_upper_speed_inner,
                    y_lower_step_inner.astype(jnp.float32),
                    y_lower_speed_inner,
                    42.0
                )

            split_condition = jnp.logical_and(
                jax.random.bernoulli(key_split, p=ENEMY_MISSILE_SPLIT_PROBABILITY),
                jnp.logical_and(did_split_upper_step != 42.0, did_split_lower_step != 42.0)
            )

            y_upper_changed, y_upper_speed_changed, y_lower_changed, y_lower_speed_changed, flag = jax.lax.cond(
                split_condition,
                lambda _: do_split(),
                lambda _: dont_split(),
                operand=None
            )

            stepped_upper_missile = jnp.array([x_upper_step, y_upper_changed, y_upper_speed_changed, flag], dtype=jnp.float32)
            stepped_lower_missile = jnp.array([x_lower_step, y_lower_changed, y_lower_speed_changed, flag], dtype=jnp.float32)

            # Kill upper if out of bounds
            stepped_upper_missile = jnp.where(
                jnp.logical_or(y_upper_changed < 44.0, y_upper_changed > 163.0),
                jnp.array([0.0, 0.0, 0.0, 187.0], dtype=jnp.float32),
                stepped_upper_missile,
            )

            # Kill lower if out of bounds
            stepped_lower_missile = jnp.where(
                jnp.logical_or(y_lower_changed < 44.0, y_lower_changed > 163.0),
                jnp.array([0.0, 0.0, 0.0, 187.0], dtype=jnp.float32),
                stepped_lower_missile
            )

            return stepped_upper_missile, stepped_lower_missile

        # Check if missile is "alive"
        upper_dead = jnp.all(missile_upper == jnp.array([0.0, 0.0, 0.0, 187.0], dtype=jnp.float32))
        lower_dead = jnp.all(missile_lower == jnp.array([0.0, 0.0, 0.0, 187.0], dtype=jnp.float32))

        updated_missile_upper, updated_missile_lower = jax.lax.cond(
            jnp.logical_and(upper_dead, lower_dead),
            lambda _: maybe_spawn(),
            lambda _: do_step(),
            operand=None
        )

        missiles = missiles.at[i].set(updated_missile_upper)
        missiles = missiles.at[i + 1].set(updated_missile_lower)

        return missiles, key

    def step_even(i, carry):
        a, b = carry
        carry = a, b, i
        return step_both_missiles(i * 2, carry)

    updated, _ = jax.lax.fori_loop(
        0, enemies.shape[0], step_even, (missile_states, rng)
    )
    return updated


@jax.jit
def player_missile_step(state: ChopperCommandState, curr_player_x, curr_player_y, action: chex.Array):
    fire = jnp.any(
        jnp.array([
            action == Action.FIRE,
            action == Action.UPRIGHTFIRE,
            action == Action.UPLEFTFIRE,
            action == Action.DOWNFIRE,
            action == Action.DOWNRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.RIGHTFIRE,
            action == Action.LEFTFIRE,
            action == Action.UPFIRE,
        ])
    )

    missile_y = curr_player_y + 6
    cooldown = jnp.maximum(state.player_missile_cooldown - 1, 0)

    def try_spawn(missiles):
        def body(i, carry):
            missiles, did_spawn = carry
            missile = missiles[i]
            free = missile[2] == 0  # direction == 0 -> inactive
            should_spawn = jnp.logical_and(free, jnp.logical_not(did_spawn))

            spawn_x = jnp.where(
                state.player_facing_direction == -1,
                curr_player_x - PLAYER_MISSILE_WIDTH,
                curr_player_x + PLAYER_SIZE[0],
            )

            new_missile = jnp.array([
                spawn_x, # x
                missile_y, # y
                state.player_facing_direction, # dir
                spawn_x # x_spawn
            ], dtype=jnp.int32)

            updated_missile = jnp.where(should_spawn, new_missile, missile)
            missiles = missiles.at[i].set(updated_missile)
            return missiles, jnp.logical_or(did_spawn, should_spawn)

        return jax.lax.fori_loop(0, missiles.shape[0], body, (missiles, False))

    def spawn_if_possible(missiles):
        def do_spawn(_):
            return try_spawn(missiles)
        def skip_spawn(_):
            return missiles, False
        return jax.lax.cond(jnp.logical_and(jnp.logical_and(fire, state.pause_timer > DEATH_PAUSE_FRAMES), cooldown == 0), do_spawn, skip_spawn, operand=None)

    def update_missile(missile):
        exists = missile[2] != 0
        new_x = missile[0] + missile[2] * MISSILE_SPEED + state.player_velocity_x

        updated = jnp.array([
            new_x,        # updated x
            missile[1],   # y stays
            missile[2],   # direction stays
            missile[3]    # x_spawn stays
        ], dtype=jnp.int32)

        chopper_pos = (WIDTH // 2) - 8 + state.local_player_offset + (state.player_velocity_x * DISTANCE_WHEN_FLYING)
        left_bound = state.player_x - chopper_pos - PLAYER_MISSILE_WIDTH
        right_bound = state.player_x + (WIDTH - chopper_pos)

        out_of_bounds = jnp.logical_or(updated[0] < left_bound, updated[0] > right_bound)
        return jnp.where(jnp.logical_and(exists, ~out_of_bounds), updated, jnp.array([0, 0, 0, 0], dtype=jnp.int32))

    updated_missiles = jax.vmap(update_missile)(state.player_missile_positions)
    updated_missiles, did_spawn = spawn_if_possible(updated_missiles)
    new_cooldown = jnp.where(did_spawn, MISSILE_COOLDOWN_FRAMES, cooldown)

    return updated_missiles, new_cooldown


@jax.jit
def player_step(
    state: ChopperCommandState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    # Bewegungsrichtung bestimmen
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


    # Ziel-Beschleunigung basierend auf Eingabe
    accel_x = jnp.where(right, ACCEL, jnp.where(left, -ACCEL, 0.0))

    # Direction player is facing
    new_player_facing_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_facing_direction))

    # Neue Geschwindigkeit berechnen und begrenzen
    velocity_x = state.player_velocity_x + accel_x
    velocity_x = jnp.clip(velocity_x, -MAX_VELOCITY, MAX_VELOCITY)

    # Falls keine Eingabe: langsamer werden (Friction)
    velocity_x = jnp.where(~(left | right), velocity_x * (1.0 - FRICTION), velocity_x)

    # Neue X-Position (global!)
    player_x = state.player_x + velocity_x

    # Y-Position berechnen (sofortige Reaktion)
    delta_y = jnp.where(up, -1, jnp.where(down, 1, 0))
    player_y = jnp.clip(state.player_y + delta_y, PLAYER_BOUNDS[1][0], PLAYER_BOUNDS[1][1])

    # "Momentum" berechnen für Offset von der Mitte aus
    new_player_offset = jnp.where(new_player_facing_direction == 1, state.local_player_offset - LOCAL_PLAYER_OFFSET_SPEED, state.local_player_offset + LOCAL_PLAYER_OFFSET_SPEED)
    new_player_offset = jnp.asarray(new_player_offset, dtype=jnp.int32)

    new_player_offset = jnp.clip(new_player_offset, -60, 60)

    return player_x, player_y, velocity_x, new_player_offset, new_player_facing_direction

def lives_step(
        player_collision: chex.Array,
        current_score: jnp.int32,
        save_lives: jnp.int32,
) -> tuple[jnp.int32, jnp.int32]:

    current_score = jnp.where(current_score == 0, 1, current_score)
    num_to_add = jnp.where(player_collision, -1, 0)

    num_to_add = jnp.where((current_score // 10000) > save_lives, num_to_add + 1, num_to_add)
    new_save_lives = jnp.where((current_score // 10000) > save_lives, save_lives + 1, save_lives)

    return num_to_add, new_save_lives


class JaxChopperCommand(JaxEnvironment[ChopperCommandState, ChopperCommandObservation, ChopperCommandInfo]):
    def __init__(self, frameskip: int = 1, reward_funcs: list[Callable] =None):
        super().__init__()
        self.frameskip = frameskip
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
        self.obs_size = 5 + MAX_CHOPPERS * 5 + MAX_PLAYER_MISSILES * 5 + 5 + 5

    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([entity.x, entity.y, entity.width, entity.height, entity.active])

    def obs_to_flat_array(self, obs: ChopperCommandObservation, enemies: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_entity_position(obs.player),
            enemies.flatten(),
            obs.enemy_missiles.flatten(),
            self.flatten_entity_position(obs.player_missile),
            obs.player_score.flatten(),
            obs.lives.flatten(),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=np.uint8,
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: ChopperCommandState) -> ChopperCommandObservation:
        # Create player (already scalar, no need for vectorization)
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
            active=jnp.array(1),  # Player is always active
        )

        # Define a function to convert enemy positions to entity format
        def convert_to_entity(pos, size):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                pos[2] != 0,  # active flag
            ])

        # Apply conversion to each type of entity using vmap

        # Enemy jets
        jets = jax.vmap(lambda pos: convert_to_entity(pos, JET_SIZE))(
            state.jet_positions
        )

        # Friendly trucks
        trucks = jax.vmap(lambda pos: convert_to_entity(pos, TRUCK_SIZE))(
            state.truck_positions
        )

        # Enemy choppers
        choppers = jax.vmap(lambda pos: convert_to_entity(pos, CHOPPER_SIZE))(
            state.chopper_positions
        )

        # Enemy missiles
        enemy_missiles = jax.vmap(lambda pos: convert_to_entity(pos, PLAYER_MISSILE_SIZE))(
            state.enemy_missile_positions
        )

        # Player missile (scalar)
        missile_pos = state.player_missile_positions
        player_missile = EntityPosition(
            x=missile_pos[0],
            y=missile_pos[1],
            width=jnp.array(PLAYER_MISSILE_SIZE[0]),
            height=jnp.array(PLAYER_MISSILE_SIZE[1]),
            active=jnp.array(missile_pos[2] != 0),
        )

        # Return observation
        return ChopperCommandObservation(
            player=player,
            trucks=trucks,
            jets=jets,
            choppers=choppers,
            enemy_missiles=enemy_missiles,
            player_missile=player_missile,
            player_score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: ChopperCommandState, all_rewards: jnp.ndarray) -> ChopperCommandInfo:
        return ChopperCommandInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: ChopperCommandState, state: ChopperCommandState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: ChopperCommandState, state: ChopperCommandState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: ChopperCommandState) -> bool:
        return state.lives < 0



    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(30)) -> Tuple[ChopperCommandObservation, ChopperCommandState]:
        """Initialize game state"""

        jet_positions, _ = initialize_enemy_positions(key)
        _, chopper_positions = initialize_enemy_positions(key)
        initial_enemy_missile_positions = jnp.full((MAX_ENEMY_MISSILES, 4), jnp.array([0, 0, 0, 187], dtype=jnp.float32))

        # FOR EXPLANATION OF FIELDS SEE ChopperCommandState CLASS
        reset_state = ChopperCommandState(
            player_x=jnp.array(PLAYER_START_X).astype(jnp.float32),             # Initial horizontal spawn position of the player in world space. Sets where the chopper starts.
            player_y=jnp.array(PLAYER_START_Y).astype(jnp.int32),               # Initial vertical spawn position of the player. Ensures the chopper appears above ground level.
            player_velocity_x=jnp.array(0).astype(jnp.float32),                 # Player starts at rest horizontally — no momentum on the first frame.
            local_player_offset=jnp.array(50).astype(jnp.float32),              # Initial screen offset. This is for the no_move_pause animation when starting the game and respawning.
            player_facing_direction=jnp.array(1).astype(jnp.int32),             # # Player begins facing right (1).
            score=jnp.array(0).astype(jnp.int32),                               # Game always starts with a score of 0.
            lives=jnp.array(3).astype(jnp.int32),                               # Standard number of starting lives.
            save_lives=jnp.array(0).astype(jnp.int32),                          # Player starts with no lives granted for every 10000 points reached
            truck_positions=initialize_truck_positions().astype(jnp.float32),   # Trucks are initialized with predefined starting positions and inactive death timers.
            jet_positions=jet_positions,                                        # Jets are initialized with predefined starting positions and inactive death timers.
            chopper_positions=chopper_positions,                                # Choppers are initialized with predefined starting positions and inactive death timers.
            enemy_missile_positions=initial_enemy_missile_positions,            # All enemy missile entries are coded as "dead": [0.0, 0.0, 0.0, 187.0], meaning no missiles are in play at start.
            player_missile_positions=jnp.zeros((MAX_PLAYER_MISSILES, 4)),       # All player missile slots are zeroed out; meaning no missiles are in play at start.
            player_missile_cooldown=jnp.array(0),                               # Cooldown timer is 0, so the player is allowed to shoot immediately.
            player_collision=jnp.array(False),                                  # Player has not collided with anything on game start.
            step_counter=jnp.array(0).astype(jnp.int32),                        # Frame counter starts from 0.
            pause_timer=jnp.array(DEATH_PAUSE_FRAMES + 2).astype(jnp.int32),    # The game starts in the no_move_pause (DEATH_PAUSE_FRAMES + 2) to allow for visual startup or intro.
            rng_key=key,                                                        # Pseudo random number generator seed.
            obs_stack=jnp.zeros((self.frame_stack_size, self.obs_size)),        # Observation stack starts empty (zeros). Used for agent state.
        )

        initial_obs = self._get_observation(reset_state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)
        reset_state = reset_state._replace(obs_stack=initial_obs)
        return initial_obs, reset_state


    @partial(jax.jit, static_argnums=(0,))
    def step(
            self, state: ChopperCommandState, action: chex.Array
    ) -> Tuple[ChopperCommandState, ChopperCommandObservation, jnp.ndarray, bool, ChopperCommandInfo]:

        previous_state = state

        #The normal_game_state is one step (e.g. one frame) of the normally ongoing game
        def get_normal_game_state(state, action):


            # --------------- POSITION UPDATES ---------------

            # Update player position
            (
                new_player_x,
                new_player_y,
                new_player_velocity_x,
                new_local_player_offset,
                new_player_facing_direction
            ) = player_step(
                state, action
            )

            # Update jet and chopper positions
            new_jet_positions, new_chopper_positions, new_rng_key = step_enemy_movement(
                state.truck_positions,
                state.jet_positions,
                state.chopper_positions,
                state.step_counter,
                state.rng_key,
                state.player_x,
            )

            #Update truck positions
            new_truck_positions = (
                step_truck_movement(
                    state.truck_positions,
                    state.player_x,
                )
            )

            # Update enemy missile positions
            new_enemy_missile_positions = enemy_missiles_step(
                new_jet_positions, new_chopper_positions, state.enemy_missile_positions, state.rng_key
            )

            # Update player missile positions
            new_player_missile_positions, new_cooldown = player_missile_step(
                state, state.player_x, state.player_y, action
            )


            # --------------- COLLISION UPDATES ---------------

            on_screen_chopper_position = (WIDTH // 2) - 8 + new_local_player_offset + (new_player_velocity_x * DISTANCE_WHEN_FLYING)

            # Check player missile collisions with jets
            (
                new_player_missile_position,
                new_jet_positions,
                new_score_jet,
                new_rng_key,
            ) = check_missile_collisions(
                new_player_missile_positions,
                new_jet_positions,
                state.score,
                new_rng_key,
                on_screen_chopper_position,
                new_player_x,
                JET_SIZE
            )

            # Check player missile collisions with choppers
            (
                new_player_missile_position,
                new_chopper_positions,
                new_score,
                new_rng_key,
            ) = check_missile_collisions(
                new_player_missile_positions,
                new_chopper_positions,
                new_score_jet,
                new_rng_key,
                on_screen_chopper_position,
                new_player_x,
                CHOPPER_SIZE
            )

            # Check player collision with jets
            player_collision_jet, player_collision_new_jet_pos = check_player_collision_entity(
                new_player_x,
                new_player_y,
                new_player_velocity_x,
                new_jet_positions,
                JET_SIZE,
                FRAMES_DEATH_ANIMATION_ENEMY,
            )
            new_jet_positions = player_collision_new_jet_pos

            # Check player collision with choppers
            player_collision_chopper, player_collision_new_chopper_pos = check_player_collision_entity(
                new_player_x,
                new_player_y,
                new_player_velocity_x,
                new_chopper_positions,
                CHOPPER_SIZE,
                FRAMES_DEATH_ANIMATION_ENEMY,
            )
            new_chopper_positions = player_collision_new_chopper_pos

            # Check player collision with trucks
            player_collision_truck, player_collision_new_truck_pos = check_player_collision_entity(
                new_player_x,
                new_player_y,
                new_player_velocity_x,
                new_truck_positions,
                TRUCK_SIZE,
                FRAMES_DEATH_ANIMATION_TRUCK,
            )
            new_truck_positions = player_collision_new_truck_pos


            # Check player collision with enemy missiles
            player_collision_enemy_missile, player_collision_new_missile_pos = check_player_collision_entity(
                new_player_x,
                new_player_y,
                new_player_velocity_x,
                new_enemy_missile_positions,
                ENEMY_MISSILE_SIZE,
                0,
            )
            new_enemy_missile_positions = player_collision_new_missile_pos


            # Check enemy missiles collision with trucks
            mis_tru_collision_truck_positions, mis_tru_collision_missile_positions = check_missile_truck_collisions(
                new_truck_positions,
                new_enemy_missile_positions,
                TRUCK_SIZE,
                ENEMY_MISSILE_SIZE
            )

            new_truck_positions = mis_tru_collision_truck_positions
            new_enemy_missile_positions = mis_tru_collision_missile_positions


            # If player collided with anything
            player_collision = jnp.logical_or(
                player_collision_jet,
                jnp.logical_or(
                    player_collision_chopper,
                    jnp.logical_or(
                        player_collision_truck,
                        player_collision_enemy_missile
                    )
                )
            )


            # --------------- DEATH UPDATES ---------------

            # Update enemy death
            new_chopper_positions = update_entity_death(new_chopper_positions, FRAMES_DEATH_ANIMATION_ENEMY, jnp.array(False))
            new_jet_positions = update_entity_death(new_jet_positions, FRAMES_DEATH_ANIMATION_ENEMY, jnp.array(False))
            new_truck_positions = update_entity_death(new_truck_positions, FRAMES_DEATH_ANIMATION_TRUCK, jnp.array(True))


            # --------------- SCORE AND LIFE UPDATES ---------------

            # Update score with collision points
            new_score = jnp.where(player_collision_jet, new_score + SCORE_PER_JET_KILL, new_score)
            new_score = jnp.where(player_collision_chopper, new_score + SCORE_PER_CHOPPER_KILL, new_score)

            # Update lives if player collides with an enemy or enemy missile and add one life for every 10000 points
            new_lives_to_add, new_save_lives = lives_step(player_collision, state.score, state.save_lives)
            new_lives = state.lives + new_lives_to_add # TODO: Make score and lives update match real game (very small detail)


            # --------------- STEP COUNT ---------------

            # Update step counter
            new_step_counter = state.step_counter + 1


            # --------------- CREATE THE NORMAL RETURNED STATE ---------------

            inner_normal_returned_state = ChopperCommandState(
                player_x=new_player_x,
                player_y=new_player_y,
                player_velocity_x=new_player_velocity_x,
                local_player_offset=new_local_player_offset,
                player_facing_direction=new_player_facing_direction,
                score=new_score,
                lives=new_lives,
                save_lives=new_save_lives,
                truck_positions=new_truck_positions,
                jet_positions=new_jet_positions,
                chopper_positions=new_chopper_positions,
                enemy_missile_positions=new_enemy_missile_positions,
                player_missile_positions=new_player_missile_position,
                player_missile_cooldown=new_cooldown,
                player_collision=player_collision,
                step_counter=new_step_counter,
                pause_timer=state.pause_timer,
                rng_key=new_rng_key,
                obs_stack=state.obs_stack,  # Include obs_stack in the state
            )

            return inner_normal_returned_state

        # The death_pause is the pause that occurs right after the chopper died
        def get_death_pause_state(state, normal_state, action):
            in_pause = jnp.logical_and(state.pause_timer <= DEATH_PAUSE_FRAMES, state.pause_timer > 0)

            # pause counter
            new_pause_timer = jnp.where(in_pause, state.pause_timer - 1, state.pause_timer)
            # Freeze pause counter if player has no lives left
            new_pause_timer = jnp.where(state.lives == 0, jnp.maximum(new_pause_timer, 1), new_pause_timer)

            # truck deaths
            temp_truck_positions = (
                step_truck_movement(
                    normal_state.truck_positions,
                    normal_state.player_x,
                )
            )

            updated_truck_death_timers = temp_truck_positions[:, 3]  # neue Timer
            new_truck_positions = state.truck_positions.at[:, 3].set(updated_truck_death_timers)

            # Enemy deaths
            temp_jet_positions, temp_chopper_positions, _ = (
                step_enemy_movement(
                    normal_state.truck_positions,
                    normal_state.jet_positions,
                    normal_state.chopper_positions,
                    normal_state.step_counter,
                    normal_state.rng_key,
                    normal_state.player_x,
                )
            )

            updated_jet_death_timers = temp_jet_positions[:, 3]  # neue Timer
            updated_chopper_death_timers = temp_chopper_positions[:, 3]  # neue Timer

            new_jet_positions_pause = state.jet_positions.at[:, 3].set(updated_jet_death_timers)
            new_chopper_positions_pause = state.chopper_positions.at[:, 3].set(updated_chopper_death_timers)


            new_player_missile_positions, new_cooldown = player_missile_step(
                normal_state, normal_state.player_x, normal_state.player_y, action
            )

            new_player_missile_positions = new_player_missile_positions.astype(jnp.float32)

            new_enemy_missile_positions = jnp.full((MAX_ENEMY_MISSILES, 4), jnp.array([0, 0, 0, 187], dtype=jnp.float32))

            # Things that have to be updated in the pause
            # truck_positions[i][3]     - this is required for rendering truck death animation while paused
            # jet_positions[i][3]       - this is required for rendering jet death animation while paused
            # chopper_positions[i][3]   - this is required for rendering chopper death animation while paused
            # enemy_missile_positions   - this is required to delete all enemy missiles instantly when player dies
            # player_missile_positions  - this is required to let player missile travel while paused
            # player_missile_cooldown   - this is a dependency for player_missile_positions
            # pause_timer               - this is required for ending the pause
            paused_state = state._replace(
                truck_positions=new_truck_positions,
                jet_positions=new_jet_positions_pause,
                chopper_positions=new_chopper_positions_pause,
                enemy_missile_positions=new_enemy_missile_positions,
                player_missile_positions=new_player_missile_positions,
                player_missile_cooldown=new_cooldown,
                pause_timer=new_pause_timer,
            )

            return paused_state

        # The no_move_pause is the pause between the pause that occurs instantly after death and the first input
        def get_no_move_pause_state(state, action):
            no_input = jnp.isin(action, Action.NOOP)

            on_screen_position = (WIDTH // 2) - 8 + state.local_player_offset + (state.player_velocity_x * DISTANCE_WHEN_FLYING)
            chopper_at_desired_point = on_screen_position <= ALLOW_MOVE_OFFSET

            # Things that have to be updated in the no_move_pause
            # local_player_offset - this is required for moving the chopper to the left of the screen, since we are not calling player_step anymore
            # pause_timer         - this is required for keeping the game paused until chopper is at the desired location and an input is made
            no_move_state = state._replace(
                local_player_offset=jnp.where(chopper_at_desired_point,
                                              state.local_player_offset,
                                              state.local_player_offset - LOCAL_PLAYER_OFFSET_SPEED
                                              ),
                pause_timer=jnp.where(jnp.logical_and(chopper_at_desired_point, jnp.logical_not(no_input)),
                                      jnp.array(DEATH_PAUSE_FRAMES + 1),
                                      jnp.array(state.pause_timer) #Stay in no_move_pause
                                      )
            )

            return no_move_state


        # Calculate all possible next states
        normal_state = get_normal_game_state(state, action)
        death_pause_state = get_death_pause_state(state, normal_state, action)
        no_move_state = get_no_move_pause_state(state, action)

        # dtype-Mismatch fix
        normal_state = jax.tree.map(
            lambda new, old: new.astype(old.dtype),
            normal_state,
            state
        ) #TODO: Find the cause of needing these. This is probably just a type mismatch error in one or more of the fields of normal_state and statev

        # dtype-Mismatch fix
        death_pause_state = jax.tree.map(
            lambda new, old: new.astype(old.dtype),
            death_pause_state,
            state,
        ) #TODO: Find the cause of needing these. This is probably just a type mismatch error in one or more of the fields of normal_state and statev

        # dtype-Mismatch fix
        no_move_state = jax.tree.map(
            lambda new, old: new.astype(old.dtype),
            no_move_state,
            state
        ) #TODO: Find the cause of needing these. This is probably just a type mismatch error in one or more of the fields of normal_state and statev

        # Pick correct state
        step_state = jax.lax.cond(
            state.pause_timer <= DEATH_PAUSE_FRAMES, #Pick the death_pause if Pause was initiated
            lambda _: death_pause_state,
            lambda _: normal_state,
            operand=None
        )

        step_state = jax.lax.cond(
            state.pause_timer == DEATH_PAUSE_FRAMES + 2, #Pick the no_move_pause if no move pause was initiated
            lambda _: no_move_state,
            lambda _: step_state,
            operand=None
        )

        # Soft- und Hard-Reset-Logik
        # Soft-Reset (früher war das normal_game_step(get_prev_fields=True))
        soft_reset_state = death_pause_state
        soft_reset_state = jax.lax.cond(
            state.pause_timer == DEATH_PAUSE_FRAMES + 2,
            lambda _: no_move_state,
            lambda _: soft_reset_state,
            operand=None
        )
        # Hard-Reset von Env.reset()
        _, hard_reset_state = self.reset()

        # Merge von Todes-Timern in den Hard-Reset-Zuständen
        def merge_death_timers(reset_pos, soft_pos):
            def body(i, acc):
                new_row = acc[i].at[3].set(soft_pos[i, 3])
                return acc.at[i].set(new_row)

            return jax.lax.fori_loop(0, reset_pos.shape[0], body, reset_pos)

        merged_jet_pos = merge_death_timers(hard_reset_state.jet_positions, soft_reset_state.jet_positions)
        merged_chop_pos = merge_death_timers(hard_reset_state.chopper_positions, soft_reset_state.chopper_positions)
        merged_truck_pos = merge_death_timers(hard_reset_state.truck_positions, soft_reset_state.truck_positions)

        # Erstelle respawn_state
        respawn_state = hard_reset_state._replace(
            # hard Resets
            player_x=hard_reset_state.player_x,
            player_y=hard_reset_state.player_y,
            # soft Resets
            score=soft_reset_state.score,
            lives=soft_reset_state.lives,
            save_lives=soft_reset_state.save_lives,
            jet_positions=merged_jet_pos,
            chopper_positions=merged_chop_pos,
            truck_positions=merged_truck_pos,
        )

        # Pause-Initialisierung und Update
        just_died = jnp.logical_and(
            step_state.player_collision,
            step_state.pause_timer > DEATH_PAUSE_FRAMES
        )
        all_enemies_dead = jnp.logical_and(
            jnp.all(step_state.jet_positions == 0),
            jnp.all(step_state.chopper_positions == 0)
        )
        initial_dead_pause = jnp.logical_and(all_enemies_dead, step_state.pause_timer > DEATH_PAUSE_FRAMES)

        respawn_state = respawn_state._replace(
            jet_positions=jnp.where(
                jnp.logical_and(all_enemies_dead, step_state.pause_timer == 0),
                hard_reset_state.jet_positions,
                respawn_state.jet_positions
            ),
            chopper_positions=jnp.where(
                jnp.logical_and(all_enemies_dead, step_state.pause_timer == 0),
                hard_reset_state.chopper_positions,
                respawn_state.chopper_positions
            ),
            truck_positions=jnp.where(
                jnp.logical_and(all_enemies_dead, step_state.pause_timer == 0),
                hard_reset_state.truck_positions,
                respawn_state.truck_positions
            )
        )

        step_state = step_state._replace(
            pause_timer=jnp.where(
                jnp.logical_or(just_died, initial_dead_pause),
                jnp.array(DEATH_PAUSE_FRAMES),
                step_state.pause_timer
            )
        )

        # Weitermachen oder Respawn
        step_state = jax.lax.cond(
            state.pause_timer != 0,
            lambda _: step_state,
            lambda _: respawn_state,
            operand=None
        )

        # Observation, Reward, Done, Info
        observation = self._get_observation(step_state)
        done = self._get_done(step_state)
        env_reward = self._get_env_reward(previous_state, step_state)
        all_rewards = self._get_all_rewards(previous_state, step_state)
        info = self._get_info(step_state, all_rewards)

        # Obs-Stack aktualisieren
        new_obs_stack = jax.tree.map(
            lambda stack, obs: jnp.concatenate([stack[1:], jnp.expand_dims(obs, 0)], axis=0),
            step_state.obs_stack,
            observation
        )
        step_state = step_state._replace(obs_stack=new_obs_stack)

        return new_obs_stack, step_state, env_reward, done, info


class Renderer_AtraJaxis(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Local position of player on screen
        chopper_position = (WIDTH // 2) + state.local_player_offset + (state.player_velocity_x * DISTANCE_WHEN_FLYING) - (PLAYER_SIZE[0] // 2) # (WIDTH // 2) - 8 = Heli mittig platzieren, state.local_player_offset = ob Heli links oder rechts auf Bildschirm, state.player_velocity_x * DISTANCE_WHEN_FLYING = Bewegen von Heli richtung Mitte wenn er fliegt

        # Bildschirmmitte relativ zur Scrollrichtung des Spielers
        static_center_x_jet = (WIDTH // 2) + state.local_player_offset - (JET_SIZE[0] // 2)
        static_center_x_chopper = (WIDTH // 2) + state.local_player_offset - (CHOPPER_SIZE[0] // 2)
        static_center_x_truck = (WIDTH // 2) + state.local_player_offset - (TRUCK_SIZE[0] // 2)

        #Initialisierung
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Render Background
        frame_idx = jnp.asarray(state.local_player_offset + (-state.player_x % WIDTH), dtype=jnp.int32) #local_player_offset = ob Heli links oder rechts auf Bildschirm ist, -state.player_x % WIDTH = Scrollen vom Hintergrund
        frame_bg = aj.get_sprite_frame(SPRITE_BG, frame_idx)

        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_friendly_truck = aj.get_sprite_frame(SPRITE_FRIENDLY_TRUCK, state.step_counter)

        def render_truck(i, raster_base):
            death_timer = state.truck_positions[i][3]
            direction = state.truck_positions[i][2]

            #am Leben: direction != 0 UND death_timer > FRAMES_DEATH_ANIMATION_TRUCK
            is_alive = jnp.logical_and(direction != 0,
                                       death_timer > FRAMES_DEATH_ANIMATION_TRUCK)

            #in der Todes-Animation: direction != 0 UND 0 < death_timer <= FRAMES_DEATH_ANIMATION_TRUCK
            is_dying = jnp.logical_and(
                direction != 0,
                jnp.logical_and(death_timer <= FRAMES_DEATH_ANIMATION_TRUCK,
                                death_timer > 0)
            )

            #flicker-Phase (nur relevant wenn is_dying)
            in_flicker_on = (death_timer % TRUCK_FLICKER_RATE) < (TRUCK_FLICKER_RATE // 2)

            # Render-Logik: immer anzeigen, solange ALIVE oder (DYING & flicker_on)
            should_render = jnp.logical_or(is_alive,
                                           jnp.logical_and(is_dying,
                                                           in_flicker_on))

            truck_screen_x = state.truck_positions[i][0] - state.player_x + static_center_x_truck
            truck_screen_y = state.truck_positions[i][1]

            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    truck_screen_x,
                    truck_screen_y,
                    frame_friendly_truck,
                    flip_horizontal=(state.truck_positions[i][2] == -1),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_TRUCKS, render_truck, raster)

        # -- JET Rendering --
        frame_enemy_jet = aj.get_sprite_frame(SPRITE_ENEMY_JET, state.step_counter)

        def render_enemy_jet(i, raster_base):
            death_timer = state.jet_positions[i][3]

            should_render = jnp.logical_and(state.jet_positions[i][2] != 0, death_timer > 0)

            jet_screen_x = state.jet_positions[i][0] - state.player_x + static_center_x_jet
            jet_screen_y = state.jet_positions[i][1]

            phase0 = death_timer > (2 * FRAMES_DEATH_ANIMATION_ENEMY) // 3
            phase1 = jnp.logical_and(
                death_timer <= (2 * FRAMES_DEATH_ANIMATION_ENEMY) // 3,
                death_timer > FRAMES_DEATH_ANIMATION_ENEMY // 3
            )
            death_sprite = jnp.where(
                phase0, ENEMY_CHOPPER_DEATH_1,
                jnp.where(phase1, ENEMY_CHOPPER_DEATH_2, ENEMY_CHOPPER_DEATH_3)
            )

            def render_true(r):
                # je nach death_timer richtigen Sprite rendern
                return jax.lax.cond(
                    death_timer <= FRAMES_DEATH_ANIMATION_ENEMY,
                    # Wenn in Death-Phase
                    lambda rr: aj.render_at(
                        rr, jet_screen_x, jet_screen_y - 2, death_sprite,
                        flip_horizontal=(state.jet_positions[i][2] == -1)
                    ),
                    # Wenn jet lebt
                    lambda rr: aj.render_at(
                        rr, jet_screen_x, jet_screen_y, frame_enemy_jet,
                        flip_horizontal=(state.jet_positions[i][2] == -1)
                    ),
                    raster_base
                )

            return jax.lax.cond(
                should_render,
                render_true,
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_JETS, render_enemy_jet, raster)

        # -- CHOPPER Rendering --
        frame_enemy_chopper = aj.get_sprite_frame(SPRITE_ENEMY_HELI, state.step_counter)

        def render_enemy_chopper(i, raster_base):
            death_timer = state.chopper_positions[i][3]

            should_render = jnp.logical_and(state.chopper_positions[i][2] != 0, death_timer > 0)

            chopper_screen_x = state.chopper_positions[i][0] - state.player_x + static_center_x_chopper
            chopper_screen_y = state.chopper_positions[i][1]

            phase0 = death_timer > (2 * FRAMES_DEATH_ANIMATION_ENEMY) // 3
            phase1 = jnp.logical_and(
                death_timer <= (2 * FRAMES_DEATH_ANIMATION_ENEMY) // 3,
                death_timer > FRAMES_DEATH_ANIMATION_ENEMY // 3
            )

            death_sprite = jnp.where(
                phase0, ENEMY_CHOPPER_DEATH_1,
                jnp.where(phase1, ENEMY_CHOPPER_DEATH_2, ENEMY_CHOPPER_DEATH_3)
            )

            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    chopper_screen_x,
                    chopper_screen_y,
                    jnp.where(
                        death_timer <= FRAMES_DEATH_ANIMATION_ENEMY,
                        death_sprite,
                        frame_enemy_chopper
                    ),
                    flip_horizontal=(state.chopper_positions[i][2] == -1),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_CHOPPERS, render_enemy_chopper, raster)

        # Render enemy missiles
        frame_enemy_missile = aj.get_sprite_frame(SPRITE_ENEMY_MISSILE, state.step_counter)

        def render_enemy_missile(i, raster_base):
            should_render = state.enemy_missile_positions[i][1] > 2
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    state.enemy_missile_positions[i][0] - state.player_x + static_center_x_chopper,
                    state.enemy_missile_positions[i][1],
                    frame_enemy_missile,
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_ENEMY_MISSILES, render_enemy_missile, raster)

        #Render Scores
        def trim_leading_zeros(digits: jnp.ndarray) -> jnp.ndarray:
            is_zero = jnp.all(digits == 0)

            # finde erste Stelle die nicht 0 ist
            first_nonzero = jnp.argmax(digits != 0)

            def on_nonzero():
                # Maske: True ab erster gültiger Ziffer
                mask = jnp.arange(digits.shape[0]) >= first_nonzero
                return jnp.where(mask, digits, -1)

            def on_zero():
                return jnp.array([-1, -1, -1, -1, -1, 0], dtype=digits.dtype)

            return jax.lax.cond(is_zero, on_zero, on_nonzero)

        score_array = aj.int_to_digits(state.score, 6)
        trimmed_digits = trim_leading_zeros(score_array)

        # Nur gültige Digits rendern
        def render_digit(raster, x_offset, digit):
            return jax.lax.cond(
                digit >= 0,
                lambda d: aj.render_label(raster, x_offset, 2, jnp.array([d], dtype=jnp.int32), DIGITS, spacing=8),
                lambda _: raster,
                operand=digit
            )

        # Schrittweise rendern mit X-Verschiebung
        def render_all_digits(raster, digits, spacing=8, x_start=16):
            def body(i, rast):
                return render_digit(rast, x_start + i * spacing, digits[i])

            return jax.lax.fori_loop(0, digits.shape[0], body, raster)

        raster = render_all_digits(raster, trimmed_digits)


        # Render lives
        raster = aj.render_indicator(
            raster, 16, 10, state.lives-1, LIFE_INDICATOR, spacing=9
        )

        # Render Player
        frame_pl_heli = aj.get_sprite_frame(SPRITE_PL_CHOPPER, state.step_counter)

        death_timer = state.pause_timer
        should_render = jnp.logical_and(death_timer != 0, death_timer != 1)

        # Schwellen berechnen
        phase0_cutoff = jnp.array(PLAYER_FADE_OUT_START_THRESHOLD_0 * DEATH_PAUSE_FRAMES).astype(jnp.int32)
        phase1_cutoff = jnp.array(PLAYER_FADE_OUT_START_THRESHOLD_1 * DEATH_PAUSE_FRAMES).astype(jnp.int32)

        # Phasen bestimmen
        phase0 = death_timer > phase0_cutoff
        phase1 = jnp.logical_and(
            death_timer <= phase0_cutoff,
            death_timer > phase1_cutoff
        )

        # Entsprechenden Sprite wählen
        death_sprite = jnp.where(
            phase0, PLAYER_DEATH_1,
            jnp.where(phase1, PLAYER_DEATH_2, PLAYER_DEATH_3)
        )

        all_enemies_dead = jnp.logical_and(jnp.all(state.jet_positions == 0), jnp.all(state.chopper_positions == 0))

        # Cond. Rendern
        raster = jax.lax.cond(
            should_render,
            lambda r: aj.render_at(
                r,
                chopper_position,
                state.player_y,
                jnp.where(
                    jnp.logical_or(death_timer > DEATH_PAUSE_FRAMES, all_enemies_dead),
                    frame_pl_heli,
                    death_sprite
                ),
                flip_horizontal=(state.player_facing_direction == -1),
            ),
            lambda r: r,
            raster,
        )

        # Render player missiles
        def render_single_missile(i, raster):
            missile = state.player_missile_positions[i]  #Indexierung IN der Funktion
            missile_active = missile[2] != 0


            missile_screen_x = missile[0] - state.player_x + chopper_position
            missile_screen_y = missile[1]

            def get_pl_missile_frame():
                delta_curr_missile_spawn = jnp.abs(missile[0] - missile[4])
                index = jnp.floor_divide(delta_curr_missile_spawn, MISSILE_ANIMATION_SPEED)
                index = jnp.clip(index, 0, 15)
                return index.astype(jnp.int32)
            frame_pl_missile = aj.get_sprite_frame(SPRITE_PL_MISSILE, get_pl_missile_frame())


            return jax.lax.cond(
                missile_active,
                lambda r: aj.render_at(
                    r,
                    missile_screen_x,
                    missile_screen_y,
                    frame_pl_missile,
                    flip_horizontal=(missile[2] == -1),
                ),
                lambda r: r,
                raster,
            )

        #Render all missiles (iterate over single missile function)
        raster = jax.lax.fori_loop(
            0,
            state.player_missile_positions.shape[0],
            render_single_missile,
            raster,
        )

        #Render minimap
        raster = self.render_minimap(chopper_position, raster, state)

        return raster

    def render_minimap(self, chopper_position, raster, state):
        # Render minimap background
        raster = aj.render_at(
            raster,
            MINIMAP_POSITION_X,
            MINIMAP_POSITION_Y,
            MINIMAP_BG,
        )

        # Render minimap mountains
        def get_minimap_mountains_frame():
            return jnp.asarray(((-state.player_x // (DOWNSCALING_FACTOR_WIDTH * 7)) % 8), dtype=jnp.int32)

        frame_minimap_mountains = aj.get_sprite_frame(MINIMAP_MOUNTAINS, get_minimap_mountains_frame())
        raster = aj.render_at(
            raster,
            MINIMAP_POSITION_X,
            MINIMAP_POSITION_Y + 3,
            frame_minimap_mountains,
        )

        # Render trucks on minimap
        def render_truck_minimap(i, raster_base):
            x, y, direction, death_timer = state.truck_positions[i]

            # Nur wirklich lebende Trucks (nicht in Death-Phase) anzeigen
            is_alive = jnp.logical_and(direction != 0,
                                       death_timer > FRAMES_DEATH_ANIMATION_TRUCK)

            weird_offset = 16
            minimap_x = weird_offset + (
                    (x - state.player_x + chopper_position)
                    // DOWNSCALING_FACTOR_WIDTH // 6
            )

            should_render = jnp.logical_and(
                is_alive,
                jnp.logical_and(minimap_x >= 0, minimap_x < MINIMAP_WIDTH)
            )

            def do_render(r):
                truck_world_x = state.truck_positions[i][0]
                truck_world_y = state.truck_positions[i][1]

                # Downscaling
                minimap_x = weird_offset + (
                            (truck_world_x - state.player_x + chopper_position) // DOWNSCALING_FACTOR_WIDTH // 6)
                minimap_y = (truck_world_y // DOWNSCALING_FACTOR_HEIGHT)

                return aj.render_at(
                    r,
                    MINIMAP_POSITION_X + minimap_x,
                    MINIMAP_POSITION_Y + 1 + minimap_y,
                    MINIMAP_TRUCK
                )

            return jax.lax.cond(should_render, do_render, lambda r: r, raster_base)

        raster = jax.lax.fori_loop(0, MAX_TRUCKS, render_truck_minimap, raster)


        # Render jets on minimap
        def render_jets_minimap(i, raster_base):
            weird_offset = 16
            jet_world_x = state.jet_positions[i][0]
            minimap_x = weird_offset + (
                        (jet_world_x - state.player_x + chopper_position) // DOWNSCALING_FACTOR_WIDTH // 6)

            is_alive = state.jet_positions[i][3] > FRAMES_DEATH_ANIMATION_ENEMY

            should_render = jnp.logical_and(
                is_alive,
                jnp.logical_and(
                    minimap_x >= 0,
                    minimap_x < MINIMAP_WIDTH
                )
            )

            def do_render(r):
                jet_world_x = state.jet_positions[i][0]
                jet_world_y = state.jet_positions[i][1]

                # Downscaling
                minimap_x = weird_offset + (
                            (jet_world_x - state.player_x + chopper_position) // DOWNSCALING_FACTOR_WIDTH // 6)
                minimap_y = (jet_world_y // (DOWNSCALING_FACTOR_HEIGHT + 1))

                return aj.render_at(
                    r,
                    MINIMAP_POSITION_X + minimap_x,
                    MINIMAP_POSITION_Y + 3 + minimap_y,
                    MINIMAP_ENEMY
                )

            return jax.lax.cond(should_render, do_render, lambda r: r, raster_base)

        raster = jax.lax.fori_loop(0, MAX_JETS, render_jets_minimap, raster)


        # Render choppers on minimap
        def render_choppers_minimap(i, raster_base):
            weird_offset = 16
            chooper_world_x = state.chopper_positions[i][0]
            minimap_x = weird_offset + (
                        (chooper_world_x - state.player_x + chopper_position) // DOWNSCALING_FACTOR_WIDTH // 6)

            is_alive = state.chopper_positions[i][3] > FRAMES_DEATH_ANIMATION_ENEMY

            should_render = jnp.logical_and(
                is_alive,
                jnp.logical_and(
                    minimap_x >= 0,
                    minimap_x < MINIMAP_WIDTH
                )
            )

            def do_render(r):
                chopper_world_x = state.chopper_positions[i][0]
                chopper_world_y = state.chopper_positions[i][1]

                # Downscaling
                minimap_x = weird_offset + (
                            (chopper_world_x - state.player_x + chopper_position) // DOWNSCALING_FACTOR_WIDTH // 6)
                minimap_y = (chopper_world_y // (DOWNSCALING_FACTOR_HEIGHT + 1))

                return aj.render_at(
                    r,
                    MINIMAP_POSITION_X + minimap_x,
                    MINIMAP_POSITION_Y + 3 + minimap_y,
                    MINIMAP_ENEMY
                )

            return jax.lax.cond(should_render, do_render, lambda r: r, raster_base)

        raster = jax.lax.fori_loop(0, MAX_CHOPPERS, render_choppers_minimap, raster)


        # Render player on minimap
        raster = aj.render_at(
            raster,
            MINIMAP_POSITION_X + 16 + (chopper_position // (DOWNSCALING_FACTOR_WIDTH * 7)),
            MINIMAP_POSITION_Y + 6 + (state.player_y // (DOWNSCALING_FACTOR_HEIGHT + 7)),
            MINIMAP_PLAYER,
        )

        #Render activision logo
        raster = aj.render_at(
            raster,
            MINIMAP_POSITION_X + (MINIMAP_WIDTH - 32) // 2,
            HEIGHT - 7 - 1, #7 = Sprite Height 1=One pixel headroom
            MINIMAP_ACTIVISION_LOGO,
        )

        return raster

def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)

if __name__ == "__main__":
    # Initialize game and renderer
    game = JaxChopperCommand(frameskip=1)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()

    renderer_AtraJaxis = Renderer_AtraJaxis()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
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
                        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_obs, curr_state, reward, done, info = jitted_step(
                    curr_state, action
                )

        # render and update pygame
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1
        clock.tick(60)

    pygame.quit()
