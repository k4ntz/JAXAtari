import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class SeaquestConstants(NamedTuple):
    # Colors
    BACKGROUND_COLOR = (0, 0, 139)  # Dark blue for water
    PLAYER_COLOR = (187, 187, 53)  # Yellow for player sub
    DIVER_COLOR = (66, 72, 200)  # Pink for divers
    SHARK_DIFFICULTY_COLORS = jnp.array(
        [
            [92, 186, 92],  # Level 0: Base green
            [213, 130, 74],  # Level 1: Orange (adjusted from original ROM)
            [
                170,
                92,
                170,
            ],  # Level 2: Purple (adjusted from original ROM COLOR_KILLER_SHARK_02)
            [213, 92, 130],  # Level 3: Pink (adjusted from original ROM)
            [186, 92, 92],  # Level 4: Red (adjusted from original ROM)
        ]
    )
    ENEMY_SUB_COLOR = (170, 170, 170)  # Gray for enemy subs
    OXYGEN_BAR_COLOR = (214, 214, 214, 255)  # White for oxygen
    SCORE_COLOR = (210, 210, 64)  # Score color
    OXYGEN_TEXT_COLOR = (0, 0, 0)  # Black for oxygen text

    # Object sizes and initial positions from RAM state
    PLAYER_SIZE = (16, 11)  # Width, Height
    DIVER_SIZE = (8, 11)
    SHARK_SIZE = (8, 7)
    ENEMY_SUB_SIZE = (8, 11)
    MISSILE_SIZE = (8, 1)

    PLAYER_START_X = 76
    PLAYER_START_Y = 46

    X_BORDERS = (0, 160)
    PLAYER_BOUNDS = (21, 134), (46, 141)

    # Maximum number of objects (from MAX_NB_OBJECTS)
    MAX_DIVERS = 4
    MAX_SHARKS = 12
    MAX_SUBS = 12
    MAX_ENEMY_MISSILES = 4
    MAX_PLAYER_TORPS = 1
    MAX_SURFACE_SUBS = 1
    MAX_COLLECTED_DIVERS = 6

    # define object orientations
    FACE_LEFT = -1
    FACE_RIGHT = 1

    SPAWN_POSITIONS_Y = jnp.array([71, 95, 119, 139])  # submarines at y=69?
    SUBMARINE_Y_OFFSET = 2
    ENEMY_MISSILE_Y = jnp.array([73, 97, 121, 141])  # missile x = submarine.x + 4
    DIVER_SPAWN_POSITIONS = jnp.array([69, 93, 117, 141])

    MISSILE_SPAWN_POSITIONS = jnp.array([39, 126])  # Right, Left

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



# Game state container
class SeaquestState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0 for right, 1 for left
    oxygen: chex.Array
    divers_collected: chex.Array
    score: chex.Array
    lives: chex.Array
    spawn_state: SpawnState
    diver_positions: chex.Array  # (4, 3) array for divers
    shark_positions: (
        chex.Array
    )  # (12, 3) array for sharks - separated into 4 lanes, 3 slots per lane [left to right]
    sub_positions: (
        chex.Array
    )  # (12, 3) array for enemy subs - separated into 4 lanes, 3 slots per lane [left to right]
    enemy_missile_positions: (
        chex.Array
    )  # (4, 3) array for enemy missiles (only the front boats can shoot)
    surface_sub_position: chex.Array  # (1, 3) array for surface submarine
    player_missile_position: (
        chex.Array
    )  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    just_surfaced: chex.Array  # Flag for tracking actual surfacing moment
    successful_rescues: (
        chex.Array
    )  # Number of times the player has surfaced with all six divers
    death_counter: chex.Array  # Counter for tracking death animation
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


class SeaquestObservation(NamedTuple):
    player: PlayerEntity
    sharks: jnp.ndarray  # Shape (12, 5) - 12 sharks, each with x,y,w,h,active
    submarines: jnp.ndarray  # Shape (12, 5)
    divers: jnp.ndarray  # Shape (4, 5)
    enemy_missiles: jnp.ndarray  # Shape (4, 5)
    surface_submarine: EntityPosition
    player_missile: EntityPosition
    collected_divers: jnp.ndarray  # Number of divers collected (0-6)
    player_score: jnp.ndarray
    lives: jnp.ndarray
    oxygen_level: jnp.ndarray  # Oxygen level (0-255)

class SeaquestInfo(NamedTuple):
    difficulty: jnp.ndarray  # Current difficulty level
    successful_rescues: jnp.ndarray  # Number of successful rescues
    step_counter: jnp.ndarray  # Current step count
    all_rewards: jnp.ndarray  # All rewards for the current step


class CarryState(NamedTuple):
    missile_pos: chex.Array
    shark_pos: chex.Array
    sub_pos: chex.Array
    score: chex.Array


# RENDER CONSTANTS
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load sprites - no padding needed for background since it's already full size
    bg1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/bg/1.npy"))
    pl_sub1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/1.npy"))
    pl_sub2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/2.npy"))
    pl_sub3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/3.npy"))
    diver1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/diver/1.npy"))
    diver2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/diver/2.npy"))
    shark1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/shark/1.npy"))
    shark2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/shark/2.npy"))
    
    enemy_sub1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_sub/1.npy"))
    enemy_sub2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_sub/2.npy"))
    enemy_sub3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_sub/3.npy"))
    pl_torp = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_torp/1.npy"))
    en_torp = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_torp/1.npy"))

    # Pad player submarine sprites to match each other
    pl_sub_sprites, pl_sub_offsets = jr.pad_to_match([pl_sub1, pl_sub2, pl_sub3])
    pl_sub_offsets = jnp.array(pl_sub_offsets)
    # Pad diver sprites to match each other
    diver_sprites, diver_offsets = jr.pad_to_match([diver1, diver2])
    diver_offsets = jnp.array(diver_offsets)

    # Pad shark sprites to match each other
    shark_sprites, shark_offsets = jr.pad_to_match([shark1, shark2])
    shark_offsets = jnp.array(shark_offsets)
    
    # Create colored versions of the animated shark sprites
    # Define colors: Green, Yellow, Pink, Orange
    shark_colors = jnp.array([
        [92, 186, 92],    # Green 
        [160, 171, 79],    # Yellow  
        [198, 89, 179],  # Pink
        [198, 108, 58],   # Orange
    ], dtype=jnp.uint8)
    
    # Recolor each shark animation frame
    def recolor_shark_frame(frame, color):
        """Recolor a shark frame to the specified color"""
        # Instead of assuming (0,0,0) is transparent, let's use a more robust approach
        # We'll look for any pixel that has some non-zero values and recolor those
        
        # Create a mask for pixels that have any non-zero RGB values
        # This is more robust than assuming specific transparency values
        mask = jnp.any(frame > 0, axis=-1, keepdims=True)
        
        # Start with the original frame
        recolored = frame.copy()
        
        # Apply the new color to non-transparent pixels
        # Use the mask to only change pixels that have content
        recolored = recolored.at[:, :, 0].set(
            jnp.where(mask[:, :, 0], color[0], frame[:, :, 0])
        )
        recolored = recolored.at[:, :, 1].set(
            jnp.where(mask[:, :, 0], color[1], frame[:, :, 1])
        )
        recolored = recolored.at[:, :, 2].set(
            jnp.where(mask[:, :, 0], color[2], frame[:, :, 2])
        )
        
        # Ensure we return integer values
        return recolored.astype(jnp.uint8)
    
    # Create colored versions of each shark animation frame
    # For green sharks, use the original sprite (no recoloring needed)
    shark_green_frames = jnp.array(shark_sprites).astype(jnp.uint8)
    shark_yellow_frames = jnp.array([recolor_shark_frame(frame, shark_colors[1]) for frame in shark_sprites], dtype=jnp.uint8)
    shark_pink_frames = jnp.array([recolor_shark_frame(frame, shark_colors[2]) for frame in shark_sprites], dtype=jnp.uint8)
    shark_orange_frames = jnp.array([recolor_shark_frame(frame, shark_colors[3]) for frame in shark_sprites], dtype=jnp.uint8)

    # Pad enemy submarine sprites to match each other
    enemy_sub_sprites, enemy_sub_offsets = jr.pad_to_match([enemy_sub1, enemy_sub2, enemy_sub3])
    enemy_sub_offsets = jnp.array(enemy_sub_offsets)

    # Pad player torpedo sprites to match each other
    pl_torp_sprites = [pl_torp]

    # Pad enemy torpedo sprites to match each other
    en_torp_sprites = [en_torp]

    # Background sprite (no padding needed)
    SPRITE_BG = jnp.expand_dims(bg1, axis=0)

    # Player submarine sprites
    SPRITE_PL_SUB = jnp.concatenate(
        [
            jnp.repeat(pl_sub_sprites[0][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[1][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[2][None], 4, axis=0),
        ]
    )

    # Diver sprites
    SPRITE_DIVER = jnp.concatenate(
        [
            jnp.repeat(diver_sprites[0][None], 16, axis=0),
            jnp.repeat(diver_sprites[1][None], 4, axis=0),
        ]
    )

    # Colored shark sprites - maintain the same animation frame structure
    # Green sharks use the original sprite (no recoloring)
    SPRITE_SHARK_GREEN = jnp.concatenate(
        [
            jnp.repeat(shark_green_frames[0][None], 16, axis=0),
            jnp.repeat(shark_green_frames[1][None], 8, axis=0),
        ]
    ).astype(jnp.uint8)
    
    SPRITE_SHARK_YELLOW = jnp.concatenate(
        [
            jnp.repeat(shark_yellow_frames[0][None], 16, axis=0),
            jnp.repeat(shark_yellow_frames[1][None], 8, axis=0),
        ]
    ).astype(jnp.uint8)
    
    SPRITE_SHARK_PINK = jnp.concatenate(
        [
            jnp.repeat(shark_pink_frames[0][None], 16, axis=0),
            jnp.repeat(shark_pink_frames[1][None], 8, axis=0),
        ]
    ).astype(jnp.uint8)
    
    SPRITE_SHARK_ORANGE = jnp.concatenate(
        [
            jnp.repeat(shark_orange_frames[0][None], 16, axis=0),
            jnp.repeat(shark_orange_frames[1][None], 8, axis=0),
        ]
    ).astype(jnp.uint8)

    # Enemy submarine sprites
    SPRITE_ENEMY_SUB = jnp.concatenate(
        [
            jnp.repeat(enemy_sub_sprites[0][None], 4, axis=0),
            jnp.repeat(enemy_sub_sprites[1][None], 4, axis=0),
            jnp.repeat(enemy_sub_sprites[2][None], 4, axis=0),
        ]
    )

    DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/seaquest/digits/{}.npy"))
    LIFE_INDICATOR = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/life_indicator/1.npy"))
    DIVER_INDICATOR = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/seaquest/diver_indicator/1.npy"))

    # Player torpedo sprites
    SPRITE_PL_TORP = jnp.repeat(pl_torp_sprites[0][None], 1, axis=0)

    # Enemy torpedo sprites
    SPRITE_EN_TORP = jnp.repeat(en_torp_sprites[0][None], 1, axis=0)
    # Return all sprites and all offsets for future use
    return (
        SPRITE_BG,
        SPRITE_PL_SUB,
        SPRITE_DIVER,
        SPRITE_SHARK_GREEN,
        SPRITE_SHARK_YELLOW,
        SPRITE_SHARK_PINK,
        SPRITE_SHARK_ORANGE,
        SPRITE_ENEMY_SUB,
        SPRITE_PL_TORP,
        SPRITE_EN_TORP,
        DIGITS,
        LIFE_INDICATOR,
        DIVER_INDICATOR,
        pl_sub_offsets,
        diver_offsets,
        shark_offsets,
        enemy_sub_offsets,
    )

# Load sprites once at module level
(
    SPRITE_BG,
    SPRITE_PL_SUB,
    SPRITE_DIVER,
    SPRITE_SHARK_GREEN,
    SPRITE_SHARK_YELLOW,
    SPRITE_SHARK_PINK,
    SPRITE_SHARK_ORANGE,
    SPRITE_ENEMY_SUB,
    SPRITE_PL_TORP,
    SPRITE_EN_TORP,
    DIGITS,
    LIFE_INDICATOR,
    DIVER_INDICATOR,
    PL_SUB_OFFSETS,
    DIVER_OFFSETS,
    SHARK_OFFSETS,
    ENEMY_SUB_OFFSETS,
) = load_sprites()


def get_shark_color_index(difficulty: chex.Array) -> chex.Array:
    """
    Determine which shark color to use based on difficulty level.
    Color cycle: Green -> Yellow -> Pink -> Orange -> Green -> Yellow -> Green -> Orange -> back to start
    
    Args:
        difficulty: Current difficulty level (0-7)
        
    Returns:
        Color index: 0=Green, 1=Yellow, 2=Pink, 3=Orange
    """
    # Map difficulty to color index using the specific 8-level pattern
    # Pattern: Green -> Yellow -> Pink -> Orange -> Green -> Yellow -> Green -> Orange -> back to start
    color_mapping = jnp.array([0, 1, 2, 3, 0, 1, 0, 3])  # 0=Green, 1=Yellow, 2=Pink, 3=Orange
    color_index = jnp.take(color_mapping, difficulty % 8)
    return color_index


def get_shark_sprite_by_difficulty(difficulty: chex.Array, step_counter: chex.Array):
    """
    Get the appropriate shark sprite based on difficulty level.
    
    Args:
        difficulty: Current difficulty level
        step_counter: Current step counter for animation
        
    Returns:
        Shark sprite frame for the current difficulty and animation frame
    """
    color_index = get_shark_color_index(difficulty)
    
    # Use jnp.where with conditions to select the correct sprite
    # This approach works better with JAX's JIT compilation
    shark_sprite = jnp.where(
        color_index == 0,
        SPRITE_SHARK_GREEN,
        jnp.where(
            color_index == 1,
            SPRITE_SHARK_YELLOW,
            jnp.where(
                color_index == 2,
                SPRITE_SHARK_PINK,
                SPRITE_SHARK_ORANGE  # color_index == 3
            )
        )
    )
    
    # Get the appropriate frame from the selected sprite
    return jr.get_sprite_frame(shark_sprite, step_counter)


class JaxSeaquest(JaxEnvironment[SeaquestState, SeaquestObservation, SeaquestInfo, SeaquestConstants]):
    def initialize_spawn_state(self) -> SpawnState:
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
            prev_sub=jnp.zeros(
                4, dtype=jnp.int32
            ),  # Track previous entity type (0 if shark, 1 if sub) -> starts at 1 since the first wave is sharks
            spawn_timers=jnp.array(
                [277, 277, 277, 277 + 60], dtype=jnp.int32
            ),  # All lanes start with same timer
            diver_array=jnp.array([1, 1, 0, 0], dtype=jnp.int32),
            lane_directions=self.consts.FIRST_WAVE_DIRS.astype(jnp.int32),  # First wave directions
        )


    def soft_reset_spawn_state(self, spawn_state: SpawnState) -> SpawnState:
        """Reset spawn_times"""
        return spawn_state._replace(
            spawn_timers=jnp.array([277, 277, 277, 277], dtype=jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
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

    @partial(jax.jit, static_argnums=(0,))
    def check_collision_batch(self, pos1, size1, pos2_array, size2):
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


    @partial(jax.jit, static_argnums=(0,))
    def check_missile_collisions(
        self,
        missile_pos: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        score: chex.Array,
        successful_rescues: chex.Array,
        spawn_state: SpawnState,
        rng_key: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
        """
        Check for collisions between player missile and enemies using a vectorized approach.
        """
        missile_rect_pos = missile_pos[:2]
        missile_active = missile_pos[2] != 0

        # --- 1. Vectorized Collision Detection ---
        all_enemies = jnp.concatenate([shark_positions, sub_positions], axis=0)
        enemy_sizes = jnp.concatenate([
            jnp.repeat(jnp.array(self.consts.SHARK_SIZE)[None, :], shark_positions.shape[0], axis=0),
            jnp.repeat(jnp.array(self.consts.ENEMY_SUB_SIZE)[None, :], sub_positions.shape[0], axis=0)
        ], axis=0)

        def check_single_enemy(enemy_pos, enemy_size):
            return self.check_collision_single(
                missile_rect_pos, self.consts.MISSILE_SIZE, enemy_pos[:2], enemy_size
            )

        all_collision_mask = jax.vmap(check_single_enemy, in_axes=(0, 0))(all_enemies, enemy_sizes)
        all_collision_mask = jnp.logical_and(missile_active, all_collision_mask)

        shark_collision_mask = all_collision_mask[:shark_positions.shape[0]]
        sub_collision_mask = all_collision_mask[shark_positions.shape[0]:]

        # --- 2. Update Game State Based on Collision Masks ---
        points_per_kill = self.calculate_kill_points(successful_rescues)
        score_increase = jnp.sum(all_collision_mask * points_per_kill)
        new_score = score + score_increase

        zeros = jnp.zeros_like(shark_positions[0])
        new_shark_positions = jnp.where(shark_collision_mask[:, None], zeros, shark_positions)
        new_sub_positions = jnp.where(sub_collision_mask[:, None], zeros, sub_positions)

        missile_was_destroyed = jnp.any(all_collision_mask)
        new_missile_pos = jnp.where(missile_was_destroyed, jnp.zeros(3), missile_pos)

        # --- 3. Update SpawnState Based on Collision Masks ---
        # The survived array is (12,), so we need to merge collision results from both
        # shark and sub slots.
        is_sub_mask_slots = jnp.repeat(spawn_state.prev_sub.astype(bool), 3) # (4,) -> (12,)
        final_collision_mask = jnp.where(is_sub_mask_slots, sub_collision_mask, shark_collision_mask)
        new_survived = jnp.where(final_collision_mask, 0, spawn_state.survived)

        # Determine which *lanes* had a collision across all 8 virtual lanes.
        lane_had_collision_8 = jnp.any(all_collision_mask.reshape(8, 3), axis=1) # Shape (8,)

        # Merge the 8-lane collision results into a 4-lane mask
        shark_lanes_hit, sub_lanes_hit = lane_had_collision_8[:4], lane_had_collision_8[4:]
        lane_had_collision_4 = jnp.where(spawn_state.prev_sub.astype(bool), sub_lanes_hit, shark_lanes_hit) # Shape (4,)

        # Update Spawn Timers for the 4 physical lanes
        new_spawn_timers = jnp.where(lane_had_collision_4, 200, spawn_state.spawn_timers)

        # Update Lane Directions for the 4 physical lanes
        rng_key, dir_rng_key = jax.random.split(rng_key)
        random_directions = jax.random.bernoulli(dir_rng_key, 0.5, (4,)).astype(jnp.int32)
        new_lane_directions = jnp.where(
            lane_had_collision_4, random_directions, spawn_state.lane_directions
        )

        new_spawn_state = spawn_state._replace(
            survived=new_survived,
            spawn_timers=new_spawn_timers,
            lane_directions=new_lane_directions,
        )

        return (
            new_missile_pos, new_shark_positions, new_sub_positions,
            new_score, new_spawn_state, rng_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision(
        self,
        player_x,
        player_y,
        submarine_list,
        shark_list,
        surface_sub_pos,
        enemy_projectile_list,
        score,
        successful_rescues,
    ) -> Tuple[chex.Array, chex.Array]:
        # check if the player has collided with any of the three given lists
        # the player is a 16x11 rectangle
        # the submarine is a 8x11 rectangle
        # the shark is a 8x7 rectangle
        # the missile is a 8x1 rectangle
        # the surface submarine is 8x11 as well

        # check if the player has collided with any of the submarines
        submarine_collisions = jnp.any(
            self.check_collision_batch(
                jnp.array([player_x, player_y]), self.consts.PLAYER_SIZE, submarine_list, self.consts.ENEMY_SUB_SIZE
            )
        )

        # check if the player has collided with any of the sharks
        shark_collisions = jnp.any(
            self.check_collision_batch(
                jnp.array([player_x, player_y]), self.consts.PLAYER_SIZE, shark_list, self.consts.SHARK_SIZE
            )
        )

        # check if the player collided with the surface submarine
        surface_collision = self.check_collision_single(
            jnp.array([player_x, player_y]),
            self.consts.PLAYER_SIZE,
            surface_sub_pos,
            self.consts.ENEMY_SUB_SIZE
        )

        # check if the player has collided with any of the enemy projectiles
        missile_collisions = jnp.any(
            self.check_collision_batch(
                jnp.array([player_x, player_y]),
                self.consts.PLAYER_SIZE,
                enemy_projectile_list,
                self.consts.MISSILE_SIZE
            )
        )

        # Calculate points for collisions.
        # When colliding with a shark or submarine the player gains points similar to killing the object
        collision_points = jnp.where(
            shark_collisions,
            self.calculate_kill_points(successful_rescues),
            jnp.where(
                submarine_collisions,
                self.calculate_kill_points(successful_rescues),
                jnp.where(surface_collision, self.calculate_kill_points(successful_rescues), 0),
            ),
        )

        return (
            jnp.any(
                jnp.array(
                    [
                        submarine_collisions,
                        shark_collisions,
                        missile_collisions,
                        surface_collision,
                    ]
                )
            ),
            collision_points,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_spawn_position(self, moving_left: chex.Array, slot: chex.Array) -> chex.Array:
        """Get spawn position based on movement direction and slot number"""
        base_y = jnp.array(self.consts.SPAWN_POSITIONS_Y[slot])
        x_pos = jnp.where(
            moving_left,
            jnp.array(165, dtype=jnp.int32),  # Start right if moving left
            jnp.array(0, dtype=jnp.int32),
        )  # Start left if moving right
        direction = jnp.where(moving_left, -1, 1)  # -1 for left, 1 for right
        return jnp.array([x_pos, base_y, direction], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def is_slot_empty(self, pos: chex.Array) -> chex.Array:
        """Check if a position slot is empty (0,0,ß)"""
        return pos[2] == 0

    @partial(jax.jit, static_argnums=(0,))
    def get_front_entity(self, i, lane_positions):
        # check on the first submarine in the lane which direction they are going
        direction = lane_positions[0][2]

        direction = jnp.where(
            lane_positions[0][2] == 0,
            jnp.where(
                lane_positions[1][2] == 0, lane_positions[2][2], lane_positions[1][2]
            ),
            lane_positions[0][2],
        )

        # if direction is 1, go from right to left until an active entity is found
        # if direction is -1, go from left to right until an active entity is found
        front_entity = jnp.where(
            direction == -1,
            jnp.where(
                lane_positions[0][2] != 0,
                lane_positions[0],
                jnp.where(
                    lane_positions[1][2] != 0,
                    lane_positions[1],
                    jnp.where(lane_positions[2][2] != 0, lane_positions[2], jnp.zeros(3)),
                ),
            ),
            jnp.where(
                lane_positions[2][2] != 0,
                lane_positions[2],
                jnp.where(
                    lane_positions[1][2] != 0,
                    lane_positions[1],
                    jnp.where(lane_positions[0][2] != 0, lane_positions[0], jnp.zeros(3)),
                ),
            ),
        )

        return front_entity

    @partial(jax.jit, static_argnums=(0,))
    def get_pattern_for_difficulty(
        self, current_pattern: chex.Array, moving_left: chex.Array
    ) -> chex.Array:
        """Returns spawn pattern based on the lane's current wave/pattern number

        Pattern meanings:
        0: Single enemy (initial pattern)
        1: Two adjacent enemies
        2: Two enemies with gap
        3: Three enemies in a row
        """
        # Basic pattern arrays for different formations
        PATTERNS = jnp.array(
            [
                [0, 0, 1],  # wave 0: Single enemy
                [0, 1, 1],  # wave 1: Two adjacent
                [1, 0, 1],  # wave 2: Two with gap
                [1, 1, 1],  # wave 3: Three in row
            ]
        )

        # Reverse pattern if moving left
        base_pattern = PATTERNS[current_pattern]

        return base_pattern

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_spawns(
        self,
        spawn_state: SpawnState,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        diver_positions: chex.Array,
        step_counter: chex.Array,
        rng: chex.PRNGKey = None,
    ) -> Tuple[SpawnState, chex.Array, chex.Array, chex.PRNGKey]:
        """Update enemy spawns using pattern-based system matching original game.
        Args:
            spawn_state: Current spawn state
            shark_positions: Current shark positions
            sub_positions: Current submarine positions
            diver_positions: Current diver positions
            step_counter: Current step counter
            rng: Optional random key for direction randomization

        Returns:
            Tuple of updated spawn state, shark positions, sub positions, and updated RNG key
        """


        new_spawn_timers = jnp.where(
            spawn_state.spawn_timers > 0,
            spawn_state.spawn_timers - 1,
            spawn_state.spawn_timers,
        )
        new_state = spawn_state._replace(spawn_timers=new_spawn_timers)

        # --- START of new vectorized calculation ---
        # 1. Vectorized check for empty lanes across all 4 lanes
        sharks_active = shark_positions.reshape(4, 3, 3)[:, :, 2] != 0
        subs_active = sub_positions.reshape(4, 3, 3)[:, :, 2] != 0
        all_lanes_empty = jnp.all(~sharks_active & ~subs_active, axis=1)  # Shape (4,)

        # 2. Vectorized check for entities that still need to be spawned
        to_be_spawned_lanes = spawn_state.to_be_spawned.reshape(4, 3)
        any_to_be_spawned = jnp.any(to_be_spawned_lanes != 0, axis=1)  # Shape (4,)

        # 3. Combine conditions to create a mask of all lanes that need an update
        all_lanes_need_update = jnp.logical_or(all_lanes_empty, any_to_be_spawned) # Shape (4,)
        # --- END of new vectorized calculation ---

        # The scan_lanes function is now much simpler
        def scan_lanes(carry, lane_idx):
            curr_state, curr_shark_positions, curr_sub_positions, curr_diver_positions, curr_rng = carry

            # Use the pre-computed mask to check if this lane needs an update
            needs_update = all_lanes_need_update[lane_idx]

            # The rest of the function proceeds as before
            new_carry = jax.lax.cond(
                needs_update,
                lambda x: process_lane(lane_idx, x), # process_lane is unchanged
                lambda x: x,
                (curr_state, curr_shark_positions, curr_sub_positions, curr_diver_positions, curr_rng),
            )

            return new_carry, None

        def initialize_new_spawn_cycle(i, carry):
            spawn_state, shark_positions, sub_positions, diver_positions, rng = carry

            # Split RNG key for this lane
            rng, lane_rng = jax.random.split(rng)

            # Get survived status for this lane (3 slots)
            lane_survived = jax.lax.dynamic_slice(spawn_state.survived, (i * 3,), (3,))

            # Update the difficulty patterns for this lane
            left_over = jnp.any(lane_survived)
            clipped_difficulty = spawn_state.difficulty % 8
            # Update spawn state
            lane_specific_pattern = jnp.where(
                jnp.logical_not(left_over),  # Only update if all destroyed
                jnp.where(
                    clipped_difficulty < 2,
                    0,
                    jnp.where(
                        clipped_difficulty < 4,
                        1,
                        jnp.where(
                            clipped_difficulty < 6,
                            2,
                            jnp.where(clipped_difficulty < 8, 3, 0),
                        ),
                    ),
                ),
                spawn_state.lane_dependent_pattern[i],
            )

            # Check if there's an active diver in this lane
            active_diver = diver_positions[i][2] != 0
            diver_direction = diver_positions[i][2]

            # If there's an active diver, use its direction, otherwise randomize
            moving_left = jnp.where(
                active_diver,
                diver_direction == -1,  # Use diver's direction if active
                spawn_state.lane_directions[i] == 1  # Otherwise use current lane direction
            )

            # get the spawn pattern for this lane
            # Check if this slot had something survive last time (if yes, we have to overwrite the current_pattern)
            current_pattern = jnp.where(
                left_over,
                lane_survived,
                self.get_pattern_for_difficulty(lane_specific_pattern, moving_left),
            )

            # make sure that in the current pattern all entries are positive (i.e. abs() on all values)
            current_pattern = jnp.abs(current_pattern)

            # in case we are going left, flip the pattern
            current_pattern = jnp.where(
                moving_left, -jnp.flip(current_pattern), current_pattern
            )

            # check if this should be a submarine or a shark
            is_sub = jnp.logical_and(left_over, jnp.logical_not(spawn_state.prev_sub[i]))

            # set the positions for the first enemy in the wave (dependent on the direction this is either the first or the last slot)
            first_slot = jnp.where(moving_left, 0, 2)

            base_pos = self.get_spawn_position(moving_left, jnp.array(i))
            # spawn the first enemy in the wave
            new_shark_positions = jnp.where(
                is_sub,
                shark_positions,
                shark_positions.at[(i * 3 + first_slot)].set(base_pos),
            )

            new_sub_positions = jnp.where(
                is_sub, sub_positions.at[(i * 3 + first_slot)].set(base_pos), sub_positions
            )

            # wipe the survived status for this lane (since we are starting a new wave)
            indices = jnp.array([i * 3, i * 3 + 1, i * 3 + 2])
            new_survived_full = spawn_state.survived.at[indices].set(
                jnp.zeros(3, dtype=jnp.int32)
            )

            # Set moving_left to the opposite of moving_left when determining which slot to clear in to_be_spawned
            new_to_be_spawned = current_pattern.at[jnp.where(moving_left, 0, 2)].set(0)

            # Update the full to_be_spawned array for this lane
            new_full_to_be_spawned = spawn_state.to_be_spawned.at[indices].set(
                new_to_be_spawned
            )

            new_spawn_state = SpawnState(
                difficulty=spawn_state.difficulty,
                lane_dependent_pattern=spawn_state.lane_dependent_pattern.at[i].set(
                    lane_specific_pattern
                ),
                to_be_spawned=new_full_to_be_spawned,
                survived=new_survived_full,
                prev_sub=spawn_state.prev_sub.at[i].set(is_sub),
                spawn_timers=spawn_state.spawn_timers.at[i].set(200),
                diver_array=spawn_state.diver_array,
                lane_directions=spawn_state.lane_directions,
            )

            return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions, rng

        # Modified continue_spawn_cycle to handle RNG
        def continue_spawn_cycle(i: int, carry):
            spawn_state, shark_positions, sub_positions, diver_positions, rng = carry

            # Rest of function remains the same, just pass along the RNG
            # get the relevant missing entities for this lane from the to_be_spawned array
            relevant_to_be_spawned = jax.lax.dynamic_slice(
                spawn_state.to_be_spawned, (i * 3,), (3,)
            )

            # check in which direction we are moving by finding the first non-zero value in the missing_entities array
            moving_left = jnp.where(
                relevant_to_be_spawned[0] == 0,
                jnp.where(
                    relevant_to_be_spawned[1] == 0,
                    jnp.where(relevant_to_be_spawned[2] == -1, True, False),
                    jnp.where(relevant_to_be_spawned[1] == -1, True, False),
                ),
                jnp.where(relevant_to_be_spawned[0] == -1, True, False),
            )

            # Find the index of the first non-zero value based on direction
            def scan_right_to_left(j, val):
                return jnp.where(relevant_to_be_spawned[2 - j] != 0, 2 - j, val)

            def scan_left_to_right(j, val):
                return jnp.where(relevant_to_be_spawned[j] != 0, j, val)

            # Use fori_loop to scan array in appropriate direction
            spawn_idx = jax.lax.cond(
                moving_left,
                lambda _: jax.lax.fori_loop(0, 3, scan_left_to_right, -1),
                lambda _: jax.lax.fori_loop(0, 3, scan_right_to_left, -1),
                operand=None,
            )

            spawn_idx = spawn_idx.astype(jnp.int32)

            # Get reference x position from neighboring entity
            # For moving right, look at entity to the right (spawn_idx + 1)
            # For moving left, look at entity to the left (spawn_idx - 1)
            reference_idx = jnp.where(moving_left, spawn_idx - 1, spawn_idx + 1)
            reference_idx = reference_idx.astype(jnp.int32)
            base_idx = i * 3  # Base index for this lane's entities

            # Get position from either shark or sub position arrays
            # We'll need to check both since we don't know which type exists
            reference_shark_pos = shark_positions[base_idx + reference_idx]
            reference_sub_pos = sub_positions[base_idx + reference_idx]

            # Use whichever position is non-zero (active)
            reference_x = jnp.where(
                reference_shark_pos[0] != 0, reference_shark_pos[0], reference_sub_pos[0]
            )

            edge_case = reference_x == 0
            # Edge Case: third option exists for the pattern 1 0 1, then check the next entity
            edge_case_reference_idx = jnp.where(moving_left, spawn_idx - 2, spawn_idx + 2)

            edge_case_reference_idx = edge_case_reference_idx.astype(jnp.int32)

            reference_x = jnp.where(
                edge_case,
                jnp.where(
                    shark_positions[base_idx + edge_case_reference_idx][0] != 0,
                    shark_positions[base_idx + edge_case_reference_idx][0],
                    sub_positions[base_idx + edge_case_reference_idx][0],
                ),
                reference_x,
            )

            # Get base spawn position for this lane
            base_spawn_pos = self.get_spawn_position(moving_left, jnp.array(i))

            # check if the base spawn position x is 16 / 32 pixels away from the reference x position (depending on the edge case pattern)
            # if yes, spawn the entity, if no, do nothing
            offset = jnp.where(edge_case, 32, 16)
            should_spawn = jnp.abs(base_spawn_pos[0] - reference_x) >= offset

            # in case reference_x is still 0 (happens in case the player destroyed the first entity in the wave), we just instantly spawn the entity
            should_spawn = jnp.where(reference_x == 0, True, should_spawn)

            spawn_pos = jnp.where(should_spawn, base_spawn_pos, jnp.zeros(3))

            # Update positions based on enemy type
            new_shark_positions = shark_positions.at[base_idx + spawn_idx].set(
                jnp.where(
                    jnp.logical_not(spawn_state.prev_sub[i]),
                    spawn_pos,
                    shark_positions[base_idx + spawn_idx],
                )
            )
            new_sub_positions = sub_positions.at[base_idx + spawn_idx].set(
                jnp.where(
                    spawn_state.prev_sub[i], spawn_pos, sub_positions[base_idx + spawn_idx]
                )
            )

            # Update the to_be_spawned array
            new_to_be_spawned = spawn_state.to_be_spawned.at[base_idx + spawn_idx].set(
                jnp.where(
                    should_spawn,
                    jnp.array(0),  # Single value
                    spawn_state.to_be_spawned[base_idx + spawn_idx],
                )
            )

            # Then create the new spawn state with the updated array
            new_spawn_state = SpawnState(
                difficulty=spawn_state.difficulty,
                lane_dependent_pattern=spawn_state.lane_dependent_pattern,
                to_be_spawned=new_to_be_spawned,
                survived=spawn_state.survived,
                prev_sub=spawn_state.prev_sub,
                spawn_timers=spawn_state.spawn_timers,
                diver_array=spawn_state.diver_array,
                lane_directions=spawn_state.lane_directions,
            )

            return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions, rng

        # Modified process_lane to handle RNG
        def process_lane(i, carry):
            loc_spawn_state, shark_positions, sub_positions, diver_positions, rng = carry
            base_idx = i * 3  # Base index for this lane's slots

            # determine if we need to initialize a new pattern or keep spawning for the current one
            # do this by checking in the relevant part of the to_be_spawned array if there are still 1s
            relevant_to_be_spawned = jax.lax.dynamic_slice(
                spawn_state.to_be_spawned, (base_idx,), (3,)
            )

            # if there are still 1s in the relevant part of the to_be_spawned array, keep spawning
            keep_spawning = jnp.any(relevant_to_be_spawned)

            # check the lane spawn timer
            lane_timer = spawn_state.spawn_timers[i]

            base_idx = i * 3
            # Get the sharks and subs for the current lane `i`
            lane_sharks = jax.lax.dynamic_slice(shark_positions, (base_idx, 0), (3, 3))
            lane_subs = jax.lax.dynamic_slice(sub_positions, (base_idx, 0), (3, 3))

            # Vectorized check for active entities in the lane
            sharks_active = lane_sharks[:, 2] != 0
            subs_active = lane_subs[:, 2] != 0
            lane_empty = jnp.all(~sharks_active & ~subs_active)

            # if the lane timer is unequal to 0, continue_spawn_cycle may still be called but initialize_new_spawn_cycle should not be called
            allow_new_initialization = jnp.logical_and(lane_timer == 0, lane_empty)

            def handle_no_spawning(x):
                spawn_state, shark_positions, sub_positions, diver_positions, rng = x
                return jax.lax.cond(
                    allow_new_initialization,
                    lambda y: initialize_new_spawn_cycle(i, y),
                    lambda y: (y[0], y[1], y[2], y[3], y[4]),  # Return unchanged state
                    (spawn_state, shark_positions, sub_positions, diver_positions, rng),
                )

            new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_rng = jax.lax.cond(
                keep_spawning,
                lambda x: continue_spawn_cycle(i, x),
                handle_no_spawning,
                (loc_spawn_state, shark_positions, sub_positions, diver_positions, rng),
            )

            return new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_rng

        # Modify lane_needs_update to work with the rest of the function
        def lane_needs_update(i, spawn_state, shark_positions, sub_positions):
            base_idx = i * 3  # Base index for this lane's slots

            # get how many entities in this lane are inactive
            lane_empty = jnp.all(
                jnp.array(
                    [
                        jnp.logical_and(
                            self.is_slot_empty(shark_positions[base_idx + j]),
                            self.is_slot_empty(sub_positions[base_idx + j]),
                        )
                        for j in range(3)
                    ]
                )
            )

            # check if the to_be_spawned array has any 1s in the relevant part
            relevant_to_be_spawned = jax.lax.dynamic_slice(
                spawn_state.to_be_spawned, (base_idx,), (3,)
            )

            return jnp.logical_or(lane_empty, jnp.any(relevant_to_be_spawned))

        # Replace the manual loop with lax.scan
        lane_indices = jnp.arange(4)
        (final_state, final_shark_positions, final_sub_positions, final_diver_positions, final_rng), _ = jax.lax.scan(
            scan_lanes,
            (new_state, shark_positions, sub_positions, diver_positions, rng if rng is not None else jax.random.PRNGKey(42)),
            lane_indices
        )

        return final_state, final_shark_positions, final_sub_positions, final_rng

    @partial(jax.jit, static_argnums=(0,))
    def step_enemy_movement(
        self,
        spawn_state: SpawnState,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        step_counter: chex.Array,
        rng: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
        """Update enemy positions based on their patterns"""
        # Split RNG key for direction randomization
        rng, direction_rng = jax.random.split(rng)

        def get_shark_offset(step_counter):
            """Calculates the vertical sinusoidal-like offset for sharks."""
            phase = step_counter // 4
            cycle_position = phase % 32
            raw_offset = jnp.where(
                cycle_position < 16,
                cycle_position // 2,
                7 - (cycle_position - 16) // 2,
            )
            return raw_offset - 4

        def calculate_movement_speed(step_counter, difficulty):
            """
            Calculates movement speed based on difficulty. This function is vectorized
            and uses jnp.select for efficient conditional logic.
            """
            # Ensure difficulty is non-negative and wraps at 256 for consistent logic
            safe_difficulty = jnp.maximum(0, difficulty % 256)

            # --- Speed for difficulties 0-9 ---
            diff_lt_10 = safe_difficulty < 10
            cycle_pos = step_counter % 12
            # Movement probabilities for difficulties 0-9
            should_move_patterns = jnp.array([
                (cycle_pos % 3) == 0,  # 33%
                (cycle_pos % 2) == 0,  # 50%
                (cycle_pos % 3) != 2,  # 67%
                (cycle_pos % 4) != 3,  # 75%
                (cycle_pos % 6) != 5,  # 83%
                cycle_pos != 11,      # 92%
            ])
            # Indices to select the correct pattern based on difficulty
            indices = jnp.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
            should_move = should_move_patterns[indices[safe_difficulty]]
            speed_for_diff_0_9 = jnp.where(should_move, 1, 0)

            # --- Speed for difficulties 10+ ---
            diff_above_threshold = jnp.maximum(0, safe_difficulty - 10)
            base_speed = 1 + (diff_above_threshold // 16)
            position_in_tier = diff_above_threshold % 16

            # Probabilities for gaining +1 speed within a tier
            higher_speed_patterns = jnp.array([
                (step_counter % 16) == 0, # 6.25%
                (step_counter % 8) == 0,  # 12.5%
                (step_counter % 4) == 0,  # 25%
                (step_counter % 2) == 0,  # 50%
                (step_counter % 4) != 0,  # 75%
                (step_counter % 8) != 0,  # 87.5%
                (step_counter % 16) != 0, # 93.75%
            ])
            # Indices to select the correct probability pattern
            tier_indices = jnp.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6])
            use_higher_speed = higher_speed_patterns[tier_indices[position_in_tier]]

            speed_for_diff_10_plus = jnp.where(use_higher_speed, base_speed + 1, base_speed)

            # Select speed based on difficulty bracket
            return jnp.where(diff_lt_10, speed_for_diff_0_9, speed_for_diff_10_plus)

        def move_single_enemy(pos, is_shark, difficulty, slot_idx, step_counter):
            """Moves a single enemy. This function will be vmapped."""
            is_active = jnp.logical_not(self.is_slot_empty(pos))

            movement_speed = calculate_movement_speed(step_counter, difficulty)
            velocity_x = pos[2] * movement_speed # pos[2] is direction (-1 or 1)

            # Use modulo 4 to map 8 virtual lanes to 4 physical Y-positions
            lane_idx = (slot_idx // 3) % 4
            base_y = self.consts.SPAWN_POSITIONS_Y[lane_idx]
            y_offset = jnp.where(
                is_shark,
                get_shark_offset(step_counter),
                -self.consts.SUBMARINE_Y_OFFSET
            )
            y_position = base_y + y_offset

            # Ensure all calculations are done with integer arithmetic
            new_x = pos[0] + velocity_x
            new_pos = jnp.array([new_x, y_position, pos[2]], dtype=pos.dtype)
            new_pos = jnp.where(is_active, new_pos, pos)

            out_of_bounds = jnp.logical_or(new_pos[0] <= -8, new_pos[0] >= 168)
            final_pos = jnp.where(out_of_bounds, jnp.zeros_like(pos), new_pos)

            return final_pos, out_of_bounds

        # 1. Combine sharks and subs into a single array for vectorized processing
        # Ensure both arrays have the same dtype before concatenation
        shark_positions_int = shark_positions.astype(jnp.int32)
        sub_positions_int = sub_positions.astype(jnp.int32)
        all_positions = jnp.concatenate([shark_positions_int, sub_positions_int], axis=0)
        is_shark_array = jnp.concatenate([jnp.ones(12, dtype=bool), jnp.zeros(12, dtype=bool)])
        all_slot_indices = jnp.arange(24)

        # 2. Apply movement to all 24 enemies in parallel using vmap
        vmap_move = jax.vmap(move_single_enemy, in_axes=(0, 0, None, 0, None))
        new_all_positions, enemies_survived_mask = vmap_move(
            all_positions, is_shark_array, spawn_state.difficulty, all_slot_indices, step_counter
        )

        # 3. Handle lane-based logic by padding the 4-lane state to an 8-lane structure
        #    and then merging the results back down.

        # Pad the original 4-lane state for comparison purposes
        old_survived_padded = jnp.pad(spawn_state.survived, (0, 12))
        lane_directions_padded = jnp.pad(spawn_state.lane_directions, (0, 4))

        # Perform logic in the temporary 8-lane structure
        num_lanes = 8 # 4 for sharks, 4 for subs
        survived_mask_lanes = enemies_survived_mask.reshape(num_lanes, 3)
        old_survived_lanes = old_survived_padded.reshape(num_lanes, 3)

        any_newly_survived_in_lane = jnp.any(
            jnp.logical_and(survived_mask_lanes, old_survived_lanes == 0), axis=1
        )

        random_directions = jax.random.bernoulli(direction_rng, 0.5, (num_lanes,)).astype(spawn_state.survived.dtype) * 2 - 1
        temp_lane_directions = jnp.where(
            any_newly_survived_in_lane,
            random_directions,
            lane_directions_padded
        )

        all_pos_lanes = all_positions.reshape(num_lanes, 3, 3)
        dir0 = all_pos_lanes[:, 0, 2]
        dir1 = all_pos_lanes[:, 1, 2]
        dir2 = all_pos_lanes[:, 2, 2]
        lane_base_direction = jnp.where(dir0 != 0, dir0, jnp.where(dir1 != 0, dir1, jnp.where(dir2 != 0, dir2, 1)))

        survived_direction_per_slot = jnp.repeat(lane_base_direction, 3, axis=0)
        temp_survived = jnp.where(
            enemies_survived_mask, survived_direction_per_slot, old_survived_padded
        )

        temp_survived_lanes = temp_survived.reshape(num_lanes, 3)
        lanes_to_flip = (lane_base_direction == -1)
        flipped_survived = jnp.flip(temp_survived_lanes, axis=1)
        temp_survived_lanes = jnp.where(
            lanes_to_flip[:, None], # Expand dims for broadcasting
            flipped_survived,
            temp_survived_lanes
        )

        # 4. Merge the 8-lane results back into the 4-lane state structure
        # `spawn_state.prev_sub` (shape 4,) tells us if a lane is sharks (0) or subs (1)
        is_sub_mask = spawn_state.prev_sub[:, None].astype(bool) # Shape: (4, 1)

        # Merge `survived` results
        shark_survived_res = temp_survived_lanes[:4]
        sub_survived_res = temp_survived_lanes[4:]
        final_survived_lanes = jnp.where(is_sub_mask, sub_survived_res, shark_survived_res)
        new_survived = final_survived_lanes.flatten() # Final shape: (12,)

        # Merge `lane_directions`
        shark_dir_res, sub_dir_res = temp_lane_directions[:4], temp_lane_directions[4:]
        new_lane_directions = jnp.where(spawn_state.prev_sub.astype(bool), sub_dir_res, shark_dir_res) # Final shape: (4,)

        # Merge updates for `diver_array` and `spawn_timers`
        any_newly_survived_sharks = any_newly_survived_in_lane[:4]
        any_newly_survived_subs = any_newly_survived_in_lane[4:]
        any_newly_survived_final = jnp.where(spawn_state.prev_sub.astype(bool), any_newly_survived_subs, any_newly_survived_sharks)

        new_diver_array = jnp.where(
            jnp.logical_and(any_newly_survived_final, spawn_state.diver_array == -1),
            1,
            spawn_state.diver_array
        )
        new_spawn_timers = jnp.where(
            any_newly_survived_final,
            200,
            spawn_state.spawn_timers
        )

        # 5. Update state with merged results and split positions back into sharks and subs
        new_spawn_state = spawn_state._replace(
            survived=new_survived,
            lane_directions=new_lane_directions,
            diver_array=new_diver_array,
            spawn_timers=new_spawn_timers
        )

        new_shark_positions, new_sub_positions = jnp.split(new_all_positions, 2, axis=0)

        # Ensure the returned positions have the same dtype as the input positions
        new_shark_positions = new_shark_positions.astype(shark_positions.dtype)
        new_sub_positions = new_sub_positions.astype(sub_positions.dtype)

        return new_shark_positions, new_sub_positions, new_spawn_state, rng
    

    @partial(jax.jit, static_argnums=(0,))
    def spawn_divers(
        self,
        spawn_state: SpawnState,
        diver_positions: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        step_counter: chex.Array,
    ) -> tuple[chex.Array, SpawnState]:
        """
        Vectorized function to spawn divers according to pattern that depends on collection state.
        """
        # --- 1. Vectorized Pre-computation and Checks (for all 4 lanes at once) ---

        # Condition: Only process lanes where the spawn timer is at the trigger value.
        timers_ready_mask = spawn_state.spawn_timers == 60  # Shape: (4,)

        # Condition: A diver must not already exist in the lane.
        diver_exists_mask = diver_positions[:, 2] != 0  # Shape: (4,)

        # Condition: The enemy lane must be empty.
        # This replaces the slow list comprehension with vectorized operations.
        sharks_active_per_lane = jnp.any(shark_positions.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        subs_active_per_lane = jnp.any(sub_positions.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        lanes_are_empty_mask = jnp.logical_not(
            jnp.logical_or(sharks_active_per_lane, subs_active_per_lane)
        ) # Shape: (4,)

        # Condition: The lane must be marked as available for spawning (value of 1).
        lanes_ready_to_spawn_mask = spawn_state.diver_array == 1  # Shape: (4,)

        # Condition: Do not spawn a diver if the next enemy in that lane is a submarine.
        prev_was_not_sub = jnp.logical_not(spawn_state.prev_sub)
        something_survived = jnp.any(spawn_state.survived.reshape(4, 3) != 0, axis=1)
        next_is_sub_mask = jnp.logical_and(prev_was_not_sub, something_survived)
        # Override: if the previous enemy was a sub, the next cannot be a sub.
        next_is_sub_mask = jnp.where(spawn_state.prev_sub, False, next_is_sub_mask)

        # --- 2. Combine All Conditions to Get Final Spawn Mask ---

        # A diver should spawn only if ALL conditions for its lane are met.
        should_spawn_mask = jnp.logical_and.reduce(
            jnp.array([
                timers_ready_mask,
                jnp.logical_not(diver_exists_mask),
                lanes_are_empty_mask,
                lanes_ready_to_spawn_mask,
                jnp.logical_not(next_is_sub_mask),
            ])
        ) # Shape: (4,)

        # --- 3. Calculate New Positions and State ---

        # Calculate spawn positions for all lanes based on their direction.
        moving_left_mask = spawn_state.lane_directions == 1
        x_positions = jnp.where(moving_left_mask, 168, 0)
        directions = jnp.where(moving_left_mask, -1, 1)

        # Create the potential new diver data for all 4 lanes.
        potential_new_divers = jnp.stack(
            [x_positions, self.consts.DIVER_SPAWN_POSITIONS, directions], axis=1
        ) # Shape: (4, 3)

        # Use the final spawn mask to decide whether to use the new diver data or keep the old.
        # The `[:, None]` broadcasts the (4,) mask to the (4, 3) diver_positions array.
        new_diver_positions = jnp.where(
            should_spawn_mask[:, None],
            potential_new_divers,
            diver_positions
        )

        # Update the diver_array: If a lane was marked with -1 (diver swam off-screen)
        # and the lane is now empty, mark it as ready for a new spawn cycle (value 1).
        spawn_next_cycle_mask = jnp.logical_and(spawn_state.diver_array == -1, lanes_are_empty_mask)
        new_diver_array = jnp.where(
            spawn_next_cycle_mask,
            1,
            spawn_state.diver_array
        )

        return new_diver_positions, spawn_state._replace(diver_array=new_diver_array)

    @partial(jax.jit, static_argnums=(0,))
    def step_diver_movement(
        self,
        diver_positions: chex.Array,
        shark_positions: chex.Array,
        state_player_x: chex.Array,
        state_player_y: chex.Array,
        state_divers_collected: chex.Array,
        spawn_state: SpawnState,
        step_counter: chex.Array,
        rng: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
        """Move divers according to their pattern and handle collisions.
        Returns updated diver positions, number of collected divers, updated spawn state, and updated RNG key.
        """
        new_diver_array = spawn_state.diver_array

        def calculate_diver_movement(step_counter, difficulty):
            """Calculate diver movement based on difficulty level.

            Args:
                step_counter: Current step counter (frame number)
                difficulty: Current difficulty level (0-255)

            Returns:
                Movement speed for the current frame (0, 1, or 2+)
                0 = no movement, 1 = normal speed, 2+ = higher speeds
            """
            # Ensure difficulty is non-negative and handle wrapping
            safe_difficulty = jnp.clip(difficulty % 256, 0, 255)

            # For difficulties 0-27, we have specific movement patterns
            is_high_difficulty = safe_difficulty >= 28

            # For difficulties 0-27, determine if we should move and use speed 1
            low_diff_should_move = determine_low_difficulty_movement(
                step_counter, safe_difficulty
            )
            low_diff_speed = jnp.where(low_diff_should_move, 1, 0)

            # For difficulties 28+, always move but with varying speed
            high_diff_speed = determine_high_difficulty_speed(step_counter, safe_difficulty)

            # Return appropriate speed based on difficulty
            return jnp.where(is_high_difficulty, high_diff_speed, low_diff_speed)

        def determine_low_difficulty_movement(step_counter, difficulty):
            """Determine if the diver should move for difficulties 0-27."""
            # Create boolean masks for each difficulty bracket
            diff_0_1 = jnp.logical_and(difficulty >= 0, difficulty <= 1)
            diff_2_3 = jnp.logical_and(difficulty >= 2, difficulty <= 3)
            diff_4_5 = jnp.logical_and(difficulty >= 4, difficulty <= 5)
            diff_6_7 = jnp.logical_and(difficulty >= 6, difficulty <= 7)
            diff_8_9 = jnp.logical_and(difficulty >= 8, difficulty <= 9)
            diff_10_11 = jnp.logical_and(difficulty >= 10, difficulty <= 11)
            diff_12_13 = jnp.logical_and(difficulty >= 12, difficulty <= 13)
            diff_14_15 = jnp.logical_and(difficulty >= 14, difficulty <= 15)
            diff_16_17 = jnp.logical_and(difficulty >= 16, difficulty <= 17)
            diff_18_19 = jnp.logical_and(difficulty >= 18, difficulty <= 19)
            diff_20_21 = jnp.logical_and(difficulty >= 20, difficulty <= 21)
            diff_22_23 = jnp.logical_and(difficulty >= 22, difficulty <= 23)
            diff_24_25 = jnp.logical_and(difficulty >= 24, difficulty <= 25)
            diff_26_27 = jnp.logical_and(difficulty >= 26, difficulty <= 27)

            # Movement patterns for each bracket based on paste.txt
            # Difficulty 0-1: Move every 5th frame (20% movement)
            move_0_1 = (step_counter % 5) == 0

            # Difficulty 2-3: Move every 4th frame (25% movement)
            move_2_3 = (step_counter % 4) == 0

            # Difficulty 4-5: Move every 3rd frame (33.3% movement)
            move_4_5 = (step_counter % 3) == 0

            # Difficulty 6-7: Move in pattern [1,0,0,1,0,1,0,0] (37.5% movement)
            cycle_6_7 = step_counter % 8
            move_6_7 = jnp.logical_or(
                cycle_6_7 == 0, jnp.logical_or(cycle_6_7 == 3, cycle_6_7 == 5)
            )

            # Difficulty 8-9: Move in pattern [1,0,1,0,1,0,0,1,0,1] (50% movement)
            cycle_8_9 = step_counter % 10
            move_8_9 = jnp.logical_or(
                jnp.logical_or(cycle_8_9 == 0, cycle_8_9 == 2),
                jnp.logical_or(cycle_8_9 == 4, cycle_8_9 == 7),
            )
            move_8_9 = jnp.logical_or(move_8_9, cycle_8_9 == 9)

            # Difficulty 10-11: Move every other frame (50% movement)
            move_10_11 = (step_counter % 2) == 0

            # Difficulty 12-13: Complex pattern with ~60% movement
            cycle_12_13 = step_counter % 8
            move_12_13 = jnp.logical_or(
                jnp.logical_or(cycle_12_13 == 0, cycle_12_13 == 2),
                jnp.logical_or(cycle_12_13 == 4, cycle_12_13 == 6),
            )
            move_12_13 = jnp.logical_or(move_12_13, cycle_12_13 == 7)

            # Difficulty 14-15: Complex pattern with ~65% movement
            cycle_14_15 = step_counter % 7
            move_14_15 = jnp.logical_or(
                jnp.logical_or(cycle_14_15 == 0, cycle_14_15 == 1),
                jnp.logical_or(cycle_14_15 == 3, cycle_14_15 == 5),
            )
            move_14_15 = jnp.logical_or(move_14_15, cycle_14_15 == 6)

            # Difficulty 16-17: Complex pattern with ~70% movement
            cycle_16_17 = step_counter % 10
            move_16_17 = jnp.logical_or(
                jnp.logical_or(cycle_16_17 == 0, cycle_16_17 == 1),
                jnp.logical_or(cycle_16_17 == 3, cycle_16_17 == 4),
            )
            move_16_17 = jnp.logical_or(
                move_16_17, jnp.logical_or(cycle_16_17 == 6, cycle_16_17 == 8)
            )
            move_16_17 = jnp.logical_or(move_16_17, cycle_16_17 == 9)

            # Difficulty 18-19: Move 3 out of 4 frames (75% movement)
            move_18_19 = (step_counter % 4) != 3

            # Difficulty 20-21: Move 4 out of 5 frames (80% movement)
            move_20_21 = (step_counter % 5) != 4

            # Difficulty 22-23: Move 7 out of 8 frames (87.5% movement)
            move_22_23 = (step_counter % 8) != 7

            # Difficulty 24-25: Move 15 out of 16 frames (93.75% movement)
            move_24_25 = (step_counter % 16) != 15

            # Difficulty 26-27: Always move (100% movement)
            move_26_27 = True

            # Combine all patterns using jnp.select which is cleaner for many conditions
            # Create condition array - only the first True condition will be used
            conditions = jnp.array(
                [
                    diff_0_1,
                    diff_2_3,
                    diff_4_5,
                    diff_6_7,
                    diff_8_9,
                    diff_10_11,
                    diff_12_13,
                    diff_14_15,
                    diff_16_17,
                    diff_18_19,
                    diff_20_21,
                    diff_22_23,
                    diff_24_25,
                    diff_26_27,
                ]
            )

            # Create corresponding values array
            values = jnp.array(
                [
                    move_0_1,
                    move_2_3,
                    move_4_5,
                    move_6_7,
                    move_8_9,
                    move_10_11,
                    move_12_13,
                    move_14_15,
                    move_16_17,
                    move_18_19,
                    move_20_21,
                    move_22_23,
                    move_24_25,
                    move_26_27,
                ]
            )

            # Select the appropriate pattern based on which condition is True
            should_move = jnp.select(conditions, values, default=False)

            return should_move

        def determine_high_difficulty_speed(step_counter, difficulty):
            """Determine the speed (1 or 2+) for difficulties 28+."""
            # Adjust difficulty to start from 0 for easier tier calculations
            diff_above_27 = difficulty - 28

            # Each 16 difficulty levels form a tier (just like in shark/submarine algorithm)
            tier = diff_above_27 // 16
            position_in_tier = diff_above_27 % 16

            # Base speed for each tier (increases by 1 for each tier)
            base_speed = tier + 1
            higher_speed = tier + 2

            # Position brackets within tier (matches the pattern observed in paste.txt)
            pos_0 = position_in_tier == 0
            pos_1_3 = jnp.logical_and(position_in_tier >= 1, position_in_tier <= 3)
            pos_4_6 = jnp.logical_and(position_in_tier >= 4, position_in_tier <= 6)
            pos_7_9 = jnp.logical_and(position_in_tier >= 7, position_in_tier <= 9)
            pos_10_12 = jnp.logical_and(position_in_tier >= 10, position_in_tier <= 12)
            pos_13_14 = jnp.logical_and(position_in_tier >= 13, position_in_tier <= 14)
            pos_15 = position_in_tier == 15

            # Determine higher speed frequency based on position in tier
            # These frequencies match the observed patterns in paste.txt
            use_higher_speed_pos_0 = (step_counter % 16) == 15  # 1 in 16 frames (6.25%)
            use_higher_speed_pos_1_3 = (step_counter % 8) == 7  # 1 in 8 frames (12.5%)
            use_higher_speed_pos_4_6 = (step_counter % 4) == 3  # 1 in 4 frames (25%)
            use_higher_speed_pos_7_9 = (step_counter % 2) == 1  # 1 in 2 frames (50%)
            use_higher_speed_pos_10_12 = (step_counter % 4) != 0  # 3 in 4 frames (75%)
            use_higher_speed_pos_13_14 = (step_counter % 8) != 0  # 7 in 8 frames (87.5%)
            use_higher_speed_pos_15 = (step_counter % 16) != 0  # 15 in 16 frames (93.75%)

            # Select the appropriate higher speed frequency based on position
            # Use jnp.select for cleaner code with multiple conditions
            position_conditions = jnp.array(
                [pos_0, pos_1_3, pos_4_6, pos_7_9, pos_10_12, pos_13_14, pos_15]
            )

            speed_values = jnp.array(
                [
                    use_higher_speed_pos_0,
                    use_higher_speed_pos_1_3,
                    use_higher_speed_pos_4_6,
                    use_higher_speed_pos_7_9,
                    use_higher_speed_pos_10_12,
                    use_higher_speed_pos_13_14,
                    use_higher_speed_pos_15,
                ]
            )

            use_higher_speed = jnp.select(position_conditions, speed_values, default=False)

            # Calculate final speed: higher_speed or base_speed
            return jnp.where(use_higher_speed, higher_speed, base_speed)

        def move_single_diver(i, carry):
            # Unpack carry state - (positions, collected_count, diver_array)
            positions, collected, diver_array = carry
            diver_pos = positions[i]

            # Only process active divers (direction != 0)
            is_active = diver_pos[2] != 0

            # Check for collision with player first if diver is active
            player_collision = jnp.logical_and(
                is_active,
                self.check_collision_single(
                    jnp.array([state_player_x, state_player_y]),
                    self.consts.PLAYER_SIZE,
                    jnp.array([diver_pos[0], diver_pos[1]]),
                    self.consts.DIVER_SIZE,
                ),
            )

            # Only collect if we haven't reached max divers
            can_collect = state_divers_collected < 6
            should_collect = jnp.logical_and(player_collision, can_collect)

            # Get the three sharks in the lane
            all_shark_lane_pos = jax.lax.dynamic_slice(shark_positions, (i * 3, 0), (3, 3))

            # Get shark in the same lane for collision check
            shark_lane_pos = self.get_front_entity(i, all_shark_lane_pos)
            shark_collision = jnp.logical_and(
                is_active,
                self.check_collision_single(
                    jnp.array([shark_lane_pos[0], shark_lane_pos[1]]),
                    self.consts.SHARK_SIZE,
                    jnp.array([diver_pos[0], diver_pos[1]]),
                    self.consts.DIVER_SIZE,
                ),
            )

            # check in which direction the shark is moving and copy the direction to the diver
            direction_of_shark = jnp.where(
                shark_lane_pos[2] == 0, diver_pos[2], shark_lane_pos[2]
            )

            # Calculate movement based on difficulty
            movement_speed = calculate_diver_movement(step_counter, spawn_state.difficulty)
            should_move = movement_speed > 0

            # Calculate movement direction (with speed factor)
            # If colliding with shark, use shark's direction/speed
            # Otherwise use diver's direction with appropriate speed factor
            movement_x = jnp.where(
                shark_collision,
                shark_lane_pos[2],  # Use shark's direction/speed
                diver_pos[2] * movement_speed,  # Apply difficulty-based speed
            )

            # Calculate new position
            new_x = jnp.where(
                shark_collision,
                diver_pos[0] + movement_x,  # Move with shark
                jnp.where(
                    should_move,
                    diver_pos[0] + movement_x,  # Move with calculated speed
                    diver_pos[0],  # Stay still
                ),
            )

            # Check bounds
            out_of_bounds = jnp.logical_or(new_x <= -8, new_x >= 170)

            # Create new position array - handle collection and bounds
            new_pos = jnp.where(
                jnp.logical_or(~is_active, jnp.logical_or(out_of_bounds, should_collect)),
                jnp.zeros(3),  # Reset if out of bounds or collected
                jnp.array([new_x, self.consts.DIVER_SPAWN_POSITIONS[i], direction_of_shark]),
            )

            # Update collection count if collected
            new_collected = collected + jnp.where(should_collect, 1, 0)

            # Update diver collection tracking - mark lane as collected when diver is collected
            updated_diver_array = diver_array.at[i].set(
                jnp.where(should_collect, 0, diver_array[i])
            )

            # if the diver went out of bounds set the entry to -1
            updated_diver_array = updated_diver_array.at[i].set(
                jnp.where(out_of_bounds, -1, updated_diver_array[i])
            )

            # Update the diver position, collection count and diver_array
            return positions.at[i].set(new_pos), new_collected, updated_diver_array

        # Update all diver positions and track collections
        initial_carry = (diver_positions, state_divers_collected, new_diver_array)
        final_positions, final_collected, final_diver_array = jax.lax.fori_loop(
            0, diver_positions.shape[0], move_single_diver, initial_carry
        )

        # Handle case where all divers are collected - set all lanes to -1
        # Apply the reset only if all divers have been collected
        reset_array = jnp.where(
            jnp.all(final_diver_array == 0),
            jnp.array([-1, -1, -1, -1], dtype=jnp.int32),  # Randomized reset array
            final_diver_array,  # Otherwise keep current state
        )

        # Create updated spawn state
        updated_spawn_state = spawn_state._replace(diver_array=reset_array)

        return final_positions, final_collected, updated_spawn_state, rng

    @partial(jax.jit, static_argnums=(0,))
    def spawn_step(
        self,
        state,
        spawn_state: SpawnState,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        diver_positions: chex.Array,
        rng_key: chex.PRNGKey,
    ) -> Tuple[SpawnState, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Main spawn handling function to be called in game step"""
        # Move existing enemies
        new_shark_positions, new_sub_positions, spawn_state_after_movement, new_key = (
            self.step_enemy_movement(
                spawn_state, shark_positions, sub_positions, state.step_counter, rng_key
            )
        )

        # Update spawns using updated spawn state
        new_spawn_state, new_shark_positions, new_sub_positions, new_key = (
            self.update_enemy_spawns(
                spawn_state_after_movement,
                new_shark_positions,
                new_sub_positions,
                diver_positions,
                state.step_counter,
                new_key,
            )
        )

        # Spawn new divers with updated tracking
        new_diver_positions, final_spawn_state = self.spawn_divers(
            new_spawn_state,
            diver_positions,
            new_shark_positions,
            new_sub_positions,
            state.step_counter,
        )

        return (
            final_spawn_state,
            new_shark_positions,
            new_sub_positions,
            new_diver_positions,
            new_key,
        )


    def surface_sub_step(self, state: SeaquestState) -> chex.Array:
        # Check direction value specifically to get scalar boolean
        sub_exists = state.surface_sub_position[2] != 0

        def spawn_sub(_):
            return jnp.array([159, 45, -1])  # Always spawns right facing left

        def move_sub(carry):
            sub_pos = carry
            new_x = jnp.where(
                state.step_counter % 4 == 0,
                sub_pos[0] - 1,  # Direction always -1
                sub_pos[0],
            )

            # Return either zeros or new position
            return jnp.where(
                jnp.logical_or(new_x < -8, sub_pos[2] == 0),
                jnp.zeros(3),
                jnp.array([new_x, 45, -1]),
            )

        # Each condition needs to be scalar
        enough_rescues = state.successful_rescues >= 2
        enough_divers = state.divers_collected >= 1
        correct_timing = jnp.logical_and(
            state.step_counter % 256 == 0, state.step_counter != 0
        )

        # check if the submarine should spawn
        should_spawn = jnp.logical_and(
            jnp.logical_and(enough_rescues, enough_divers),
            jnp.logical_and(correct_timing, ~sub_exists),
        )

        temp1 = spawn_sub(state.surface_sub_position)
        temp2 = move_sub(state.surface_sub_position)

        return jnp.where(should_spawn, temp1, temp2)

    @partial(jax.jit, static_argnums=(0,))
    def enemy_missiles_step(
        self, curr_sub_positions, curr_enemy_missile_positions, step_counter, difficulty
    ) -> chex.Array:

        def calculate_missile_speed(step_counter, difficulty):
            """JAX-compatible missile speed calculation function"""
            # Base tier size is 16 difficulty levels
            tier_size = 16

            # Determine base speed (1, 2, 3, etc.) based on difficulty tier
            base_speed = 1 + (difficulty // tier_size)

            # Calculate position within the current tier (0-15)
            position_in_tier = difficulty % tier_size

            # Special case for difficulty 0
            is_diff_0 = difficulty == 0

            # Create position bracket array for each pattern
            pos_brackets = jnp.array(
                [
                    jnp.logical_and(
                        position_in_tier >= 0, position_in_tier <= 2
                    ),  # 0-2: 6.25%
                    jnp.logical_and(
                        position_in_tier >= 3, position_in_tier <= 4
                    ),  # 3-4: 12.5%
                    jnp.logical_and(
                        position_in_tier >= 5, position_in_tier <= 6
                    ),  # 5-6: 25%
                    jnp.logical_and(
                        position_in_tier >= 7, position_in_tier <= 8
                    ),  # 7-8: 50%
                    jnp.logical_and(
                        position_in_tier >= 9, position_in_tier <= 10
                    ),  # 9-10: 75%
                    jnp.logical_and(
                        position_in_tier >= 11, position_in_tier <= 12
                    ),  # 11-12: 87.5%
                    jnp.logical_and(
                        position_in_tier >= 13, position_in_tier <= 14
                    ),  # 13-14: 93.75%
                    position_in_tier == 15,  # 15: 100%
                ]
            )

            # Create array of higher speed patterns
            higher_speed_patterns = jnp.array(
                [
                    (step_counter % 16) == 0,  # 6.25%
                    (step_counter % 8) == 0,  # 12.5%
                    (step_counter % 4) == 0,  # 25%
                    (step_counter % 2) == 0,  # 50%
                    (step_counter % 4) != 0,  # 75%
                    (step_counter % 8) != 0,  # 87.5%
                    (step_counter % 16) != 0,  # 93.75%
                    True,  # 100%
                ]
            )

            # Use jnp.select to choose the pattern
            use_higher_speed = jnp.select(
                pos_brackets, higher_speed_patterns, default=False
            )

            # Higher speed is base_speed + 1
            higher_speed = base_speed + 1

            # Handle difficulty 0 special case
            return jnp.where(
                is_diff_0, 1, jnp.where(use_higher_speed, higher_speed, base_speed)
            )

        # 1. Define a function that operates on a SINGLE missile and its corresponding lane.
        #    It no longer needs an index `i` or a `carry` argument.
        def vmapped_missile_update(missile_pos, lane_subs, lane_y_pos):
            # Get the front submarine for this specific lane
            sub_pos = self.get_front_entity(0, lane_subs) # Index 0 is fine since it only looks at the 3 subs passed in

            # Check if the missile should be spawned
            missile_exists = missile_pos[2] != 0
            should_spawn = jnp.logical_and(
                ~missile_exists,
                (sub_pos[0] >= self.consts.MISSILE_SPAWN_POSITIONS[0]) &
                (sub_pos[0] <= self.consts.MISSILE_SPAWN_POSITIONS[1])
            )

            # Calculate new missile position
            new_missile_x = sub_pos[0] + 4 * sub_pos[2]
            spawned_missile = jnp.array([new_missile_x, lane_y_pos, sub_pos[2]])
            new_missile = jnp.where(should_spawn, spawned_missile, missile_pos)

            # Move the missile if it exists
            movement_speed = calculate_missile_speed(step_counter, difficulty)
            velocity = movement_speed * new_missile[2]
            moved_missile = new_missile.at[0].add(velocity)
            new_missile = jnp.where(missile_exists, moved_missile, new_missile)

            # Check bounds and return
            is_out_of_bounds = (new_missile[0] < self.consts.X_BORDERS[0]) | (new_missile[0] > self.consts.X_BORDERS[1])
            return jnp.where(is_out_of_bounds, jnp.zeros(3), new_missile)

        # 2. Prepare the inputs for vmap
        # Reshape subs into a per-lane format: (4 lanes, 3 subs per lane, 3 coords)
        all_lane_subs = curr_sub_positions.reshape(4, 3, 3)

        # 3. Use jax.vmap to apply the update function in parallel
        new_missile_positions = jax.vmap(
            vmapped_missile_update, in_axes=(0, 0, 0) # Map over missiles, sub-lanes, and y-positions
        )(curr_enemy_missile_positions, all_lane_subs, self.consts.ENEMY_MISSILE_Y)

        return new_missile_positions

    @partial(jax.jit, static_argnums=(0,))
    def player_missile_step(
        self, state: SeaquestState, curr_player_x, curr_player_y, action: chex.Array
    ) -> chex.Array:
        # check if the player shot this frame
        fire = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                    action == Action.RIGHTFIRE,
                    action == Action.LEFTFIRE,
                    action == Action.UPFIRE,
                ]
            )
        )

        # IMPORTANT: do not change the order of this check, since the missile does not move in its first frame!!
        # also check if there is currently a missile in frame by checking if the player_missile_position is empty
        missile_exists = state.player_missile_position[2] != 0

        # if the player shot and there is no missile in frame, then we can shoot a missile
        # the missile y is the current player y position + 7
        # the missile x is either player x + 3 if facing left or player x + 13 if facing right
        new_missile = jnp.where(
            jnp.logical_and(fire, jnp.logical_not(missile_exists)),
            jnp.where(
                state.player_direction == -1,
                jnp.array([curr_player_x + 3, curr_player_y + 7, -1]),
                jnp.array([curr_player_x + 13, curr_player_y + 7, 1]),
            ),
            state.player_missile_position,
        )

        # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
        new_missile = jnp.where(
            missile_exists,
            jnp.array(
                [new_missile[0] + new_missile[2] * 5, curr_player_y + 7, new_missile[2]]
            ),
            new_missile,
        )

        # check if the new positions are still in bounds
        new_missile = jnp.where(
            new_missile[0] < self.consts.X_BORDERS[0],
            jnp.array([0, 0, 0]),
            jnp.where(new_missile[0] > self.consts.X_BORDERS[1], jnp.array([0, 0, 0]), new_missile),
        )

        return new_missile

    @partial(jax.jit, static_argnums=(0,))
    def update_oxygen(self, state, player_x, player_y, player_missile_position):
        """Update oxygen levels and handle surfacing mechanics with proper surfacing detection"""
        PLAYER_BREATHING_Y = [47, 52]  # Range where oxygen neither increases nor decreases

        # Detect actual surfacing moment
        at_surface = player_y == 46
        was_underwater = player_y > 46
        just_surfaced = jnp.logical_and(at_surface, state.just_surfaced == 0)

        # Check player state
        decrease_ox = player_y > PLAYER_BREATHING_Y[1]
        has_divers = state.divers_collected >= 0  # Changed to > 0 instead of >= 0
        has_all_divers = state.divers_collected >= 6
        needs_oxygen = state.oxygen < 64

        # Special handling for initialization state
        in_init_state = state.just_surfaced == -1
        started_diving = player_y > self.consts.PLAYER_START_Y
        filling_init_oxygen = jnp.logical_and(in_init_state, state.oxygen < 64)

        # Surfacing conditions
        increase_ox = jnp.logical_and(at_surface, needs_oxygen)
        stay_same = jnp.logical_and(
            player_y >= PLAYER_BREATHING_Y[0], player_y <= PLAYER_BREATHING_Y[1]
        )

        # Calculate new divers count before other logic
        new_divers_collected = jnp.where(
            jnp.logical_and(just_surfaced, has_divers),
            jnp.where(in_init_state, state.divers_collected, state.divers_collected - 1),
            state.divers_collected,
        )

        # Handle surfacing without divers - prevent during init
        # Only lose life if we started with no divers
        lose_life = jnp.logical_and(
            jnp.logical_and(just_surfaced, new_divers_collected < 0),
            jnp.logical_not(in_init_state),
        )

        # Handle surfacing with all divers
        should_reset = jnp.logical_and(just_surfaced, has_all_divers)

        # Update surfacing flag with consideration for remaining divers
        new_just_surfaced = jnp.where(
            in_init_state,
            jnp.where(
                jnp.logical_and(started_diving, state.oxygen >= 63),
                jnp.array(0),
                jnp.array(-1),
            ),
            jnp.where(
                was_underwater,
                jnp.array(0),
                jnp.where(at_surface, jnp.array(1), state.just_surfaced),
            ),
        )

        # Handle oxygen changes
        new_oxygen = jnp.where(
            filling_init_oxygen,
            jnp.where(state.step_counter % 2 == 0, state.oxygen + 1, state.oxygen),
            jnp.where(
                decrease_ox,
                jnp.where(state.step_counter % 32 == 0, state.oxygen - 1, state.oxygen),
                state.oxygen,
            ),
        )

        # Important: Base blocking decision on has_divers instead of still_has_divers
        can_refill = jnp.logical_and(increase_ox, has_divers)
        new_oxygen = jnp.where(
            jnp.logical_and(can_refill, jnp.logical_not(in_init_state)),
            jnp.where(
                state.oxygen < 64,
                jnp.where(state.step_counter % 2 == 0, state.oxygen + 1, state.oxygen),
                state.oxygen,
            ),
            new_oxygen,
        )

        # Increase difficulty when reaching max oxygen after surfacing
        old_difficulty = state.spawn_state.difficulty
        reached_max = jnp.logical_and(
            jnp.logical_and(new_oxygen >= 64, state.oxygen < 64),
            jnp.logical_not(in_init_state),
        )
        new_difficulty = jnp.where(reached_max, old_difficulty + 1, old_difficulty)

        new_oxygen = jnp.where(stay_same, state.oxygen, new_oxygen)

        # Use has_divers for blocking decision and combine with oxygen check
        should_block = jnp.logical_and(at_surface, needs_oxygen)

        player_x = jnp.where(should_block, state.player_x, player_x)

        player_y = jnp.where(
            should_block,
            jnp.array(46, dtype=jnp.int32),  # Force to exact surface position
            player_y,
        )

        player_missile_position = jnp.where(
            should_block, jnp.zeros(3), player_missile_position
        )

        # Prevent oxygen depletion during init
        oxygen_depleted = jnp.logical_and(
            new_oxygen <= jnp.array(0), jnp.logical_not(in_init_state)
        )

        return (
            new_oxygen,
            player_x,
            player_y,
            player_missile_position,
            oxygen_depleted,
            lose_life,
            new_divers_collected,
            should_reset,
            new_just_surfaced,
            new_difficulty,
        )

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self, state: SeaquestState, action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        # implement all the possible movement directions for the player, the mapping is:
        # anything with left in it, add -1 to the x position
        # anything with right in it, add 1 to the x position
        # anything with up in it, add -1 to the y position
        # anything with down in it, add 1 to the y position
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        player_x = jnp.where(
            right, state.player_x + 1, jnp.where(left, state.player_x - 1, state.player_x)
        )

        player_y = jnp.where(
            down, state.player_y + 1, jnp.where(up, state.player_y - 1, state.player_y)
        )

        # set the direction according to the movement
        player_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_direction))

        # perform out of bounds checks
        player_x = jnp.where(
            player_x < self.consts.PLAYER_BOUNDS[0][0],
            self.consts.PLAYER_BOUNDS[0][0],  # Clamp to min player bound
            jnp.where(
                player_x > self.consts.PLAYER_BOUNDS[0][1],
                self.consts.PLAYER_BOUNDS[0][1],  # Clamp to max player bound
                player_x,
            ),
        )

        player_y = jnp.where(
            player_y < self.consts.PLAYER_BOUNDS[1][0],
            self.consts.PLAYER_BOUNDS[1][0],
            jnp.where(player_y > self.consts.PLAYER_BOUNDS[1][1], self.consts.PLAYER_BOUNDS[1][1], player_y),
        )

        return player_x, player_y, player_direction

    @partial(jax.jit, static_argnums=(0,))
    def calculate_kill_points(self, successful_rescues: chex.Array) -> chex.Array:
        """Calculate the points awarded for killing a shark or submarine. Sharks and submarines are worth 20 points.
        The points are increased by 10 for each successful rescue with a maximum of 90."""
        base_points = 20
        max_points = 90
        additional_points = 10 * successful_rescues
        return jnp.minimum(base_points + additional_points, max_points)


    def __init__(self, consts: SeaquestConstants = None, reward_funcs: list[callable] = None):
        consts = consts or SeaquestConstants()
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
        self.obs_size = 6 + 12 * 5 + 12 * 5 + 4 * 5 + 4 * 5 + 5 + 5 + 4
        self.renderer = SeaquestRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SeaquestState) -> jnp.ndarray:
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
    def obs_to_flat_array(self, obs: SeaquestObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_entity(obs.player),
            obs.sharks.flatten().astype(jnp.int32),
            obs.submarines.flatten().astype(jnp.int32),
            obs.divers.flatten().astype(jnp.int32),
            obs.enemy_missiles.flatten().astype(jnp.int32),
            self.flatten_entity_position(obs.surface_submarine),
            self.flatten_entity_position(obs.player_missile),
            obs.collected_divers.flatten().astype(jnp.int32),
            obs.player_score.flatten().astype(jnp.int32),
            obs.lives.flatten().astype(jnp.int32),
            obs.oxygen_level.flatten().astype(jnp.int32),
        ])


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Seaquest.
        The observation contains:
        - player: PlayerEntity (x, y, o, width, height, active)
        - sharks: array of shape (12, 5) with x,y,width,height,active for each shark
        - submarines: array of shape (12, 5) with x,y,width,height,active for each submarine
        - divers: array of shape (4, 5) with x,y,width,height,active for each diver
        - enemy_missiles: array of shape (4, 5) with x,y,width,height,active for each missile
        - surface_submarine: EntityPosition (x, y, width, height, active)
        - player_missile: EntityPosition (x, y, width, height, active)
        - collected_divers: int (0-6)
        - player_score: int (0-999999)
        - lives: int (0-3)
        - oxygen_level: int (0-255)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "o": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "sharks": spaces.Box(low=0, high=160, shape=(12, 5), dtype=jnp.int32),
            "submarines": spaces.Box(low=0, high=160, shape=(12, 5), dtype=jnp.int32),
            "divers": spaces.Box(low=0, high=160, shape=(4, 5), dtype=jnp.int32),
            "enemy_missiles": spaces.Box(low=0, high=160, shape=(4, 5), dtype=jnp.int32),
            "surface_submarine": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_missile": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "collected_divers": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
            "player_score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "oxygen_level": spaces.Box(low=0, high=255, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Seaquest.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: SeaquestState) -> SeaquestObservation:
        # Create player (already scalar, no need for vectorization)
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            o=state.player_direction,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
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

        # Sharks
        sharks = jax.vmap(lambda pos: convert_to_entity(pos, self.consts.SHARK_SIZE))(
            state.shark_positions
        )

        # Submarines
        submarines = jax.vmap(lambda pos: convert_to_entity(pos, self.consts.ENEMY_SUB_SIZE))(
            state.sub_positions
        )

        # Divers
        divers = jax.vmap(lambda pos: convert_to_entity(pos, self.consts.DIVER_SIZE))(
            state.diver_positions
        )

        # Enemy missiles
        enemy_missiles = jax.vmap(lambda pos: convert_to_entity(pos, self.consts.MISSILE_SIZE))(
            state.enemy_missile_positions
        )

        # Surface submarine (scalar)
        surface_pos = state.surface_sub_position
        surface_sub = EntityPosition(
            x=surface_pos[0],  # First item of first dimension
            y=surface_pos[1],  # First item of second dimension
            width=jnp.array(self.consts.ENEMY_SUB_SIZE[0]),
            height=jnp.array(self.consts.ENEMY_SUB_SIZE[1]),
            active=jnp.array(surface_pos[2] != 0),
        )

        # Player missile (scalar)
        missile_pos = state.player_missile_position
        player_missile = EntityPosition(
            x=missile_pos[0],
            y=missile_pos[1],
            width=jnp.array(self.consts.MISSILE_SIZE[0]),
            height=jnp.array(self.consts.MISSILE_SIZE[1]),
            active=jnp.array(missile_pos[2] != 0),
        )

        # Return observation
        return SeaquestObservation(
            player=player,
            sharks=sharks,
            submarines=submarines,
            divers=divers,
            enemy_missiles=enemy_missiles,
            surface_submarine=surface_sub,
            player_missile=player_missile,
            collected_divers=state.divers_collected,
            player_score=state.score,
            lives=state.lives,
            oxygen_level=state.oxygen,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SeaquestState, all_rewards: jnp.ndarray) -> SeaquestInfo:
        return SeaquestInfo(
            successful_rescues=state.successful_rescues,
            difficulty=state.spawn_state.difficulty,
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: SeaquestState, state: SeaquestState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: SeaquestState, state: SeaquestState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SeaquestState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[SeaquestObservation, SeaquestState]:
        """Initialize game state"""
        reset_state = SeaquestState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_direction=jnp.array(0),
            oxygen=jnp.array(0),  # Full oxygen
            divers_collected=jnp.array(0),
            score=jnp.array(0),
            lives=jnp.array(3),
            spawn_state=self.initialize_spawn_state(),
            diver_positions=jnp.zeros((self.consts.MAX_DIVERS, 3)),  # 4 divers
            shark_positions=jnp.zeros((self.consts.MAX_SHARKS, 3)),
            sub_positions=jnp.zeros((self.consts.MAX_SUBS, 3)),  # x, y, direction
            enemy_missile_positions=jnp.zeros((self.consts.MAX_ENEMY_MISSILES, 3)),  # 4 missiles
            surface_sub_position=jnp.zeros(3),  # 1 surface sub
            player_missile_position=jnp.zeros(3),  # x,y,direction
            step_counter=jnp.array(0),
            just_surfaced=jnp.array(-1),
            successful_rescues=jnp.array(0),
            death_counter=jnp.array(0),
            rng_key=key,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: SeaquestState, action: chex.Array
    ) -> Tuple[SeaquestObservation, SeaquestState, float, bool, SeaquestInfo]:

        previous_state = state
        _, reset_state = self.reset(state.rng_key)

        # First handle death animation if active
        def handle_death_animation():
            # This outer conditional remains the same.
            # It decides if the animation is over or still running.
            is_animation_over = state.death_counter <= 1

            def on_animation_continue():
                # This is the original logic for when the animation is still running.
                # It correctly updates the Y-positions and the player visibility.
                shark_y_positions, _, _, _ = self.step_enemy_movement(
                    state.spawn_state,
                    state.shark_positions,
                    state.sub_positions,
                    state.step_counter,
                    state.rng_key,
                )
                new_shark_positions = state.shark_positions.at[:, 1].set(
                    shark_y_positions[:, 1]
                )
                should_hide_player = state.death_counter <= 45
                return state._replace(
                    death_counter=state.death_counter - 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    player_x=jnp.where(should_hide_player, -100, state.player_x),
                    step_counter=state.step_counter + 1,
                )

            def on_animation_over():
                # This is the new, more precise logic for when the animation ends.
                # We add a check to see if this is the absolute final life.
                is_final_life = state.lives <= 0

                def handle_game_over():
                    # If this is the final life, return the TRUE final state of the game
                    # while setting lives to -1 to trigger done=True.
                    # This is what your snapshot test needs to see.
                    return state._replace(
                        lives=state.lives - 1,
                        death_counter=0,
                    )

                def handle_stage_reset():
                    # If the player still has lives left, perform the original stage reset.
                    # This preserves the mechanic of resetting the level after losing a life.
                    return reset_state._replace(
                        lives=state.lives - 1,
                        score=state.score,
                        successful_rescues=state.successful_rescues,
                        divers_collected=jnp.maximum(state.divers_collected - 1, 0),
                        spawn_state=self.soft_reset_spawn_state(state.spawn_state),
                    )

                # Use the new nested conditional to choose the correct outcome.
                return jax.lax.cond(
                    is_final_life,
                    lambda: handle_game_over(),
                    lambda: handle_stage_reset(),
                )

            # This is the main conditional call.
            return jax.lax.cond(
                is_animation_over,
                lambda: on_animation_over(),
                lambda: on_animation_continue(),
            )

        def handle_score_freeze():
            # on scoring, the death counter will be set to -(oxygen * 2 + 16 * 6)
            # thats when we get in here, so duplicate the death animation pattern, but decrease the oxygen until its 0
            # Calculate new positions with frozen X coordinates
            shark_y_positions, _, _, _ = self.step_enemy_movement(
                state.spawn_state,
                state.shark_positions,
                state.sub_positions,
                state.step_counter,
                state.rng_key,
            )

            # Keep X positions from original state, only update Y
            new_shark_positions = state.shark_positions.at[:, 1].set(
                shark_y_positions[:, 1]
            )

            # calculate the new oxygen
            new_ox = jnp.where(
                state.death_counter % 2 == 0, state.oxygen - 1, state.oxygen
            )

            new_ox = jnp.where(new_ox <= 0, jnp.array(0), state.oxygen)

            # Return either final reset or animation frame
            return jax.lax.cond(
                state.death_counter >= -1,
                lambda _: reset_state._replace(
                    player_x=state.player_x,
                    player_y=state.player_y,
                    player_direction=state.player_direction,
                    score=state.score,
                    lives=state.lives,
                    successful_rescues=state.successful_rescues,
                    divers_collected=jnp.array(0),
                    spawn_state=self.soft_reset_spawn_state(state.spawn_state),
                    surface_sub_position=state.surface_sub_position,
                    oxygen=jnp.array(0),
                ),
                lambda _: state._replace(
                    death_counter=state.death_counter + 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    step_counter=state.step_counter + 1,
                    oxygen=new_ox,
                ),
                operand=None,
            )

        # Normal game logic starts here
        def normal_game_step():
            # First check if player should be frozen for oxygen refill
            at_surface = state.player_y == 46
            needs_oxygen = state.oxygen < 64
            should_block = jnp.logical_and(at_surface, needs_oxygen)

            # while player is frozen, keep resetting the spawn counter
            new_spawn_state = jax.lax.cond(
                should_block,
                lambda: state.spawn_state._replace(
                    spawn_timers=jnp.array([80, 80, 80, 120], dtype=jnp.int32)
                ),
                lambda: state.spawn_state,
            )

            state_updated = state._replace(spawn_state=new_spawn_state)

            # If blocked, force position and disable actions
            player_x = jnp.where(should_block, state.player_x, state.player_x)
            player_y = jnp.where(
                should_block, jnp.array(46, dtype=jnp.int32), state.player_y
            )
            action_mod = jnp.where(should_block, jnp.array(Action.NOOP), action)

            # Now calculate movement using potentially modified positions and action
            next_x, next_y, player_direction = self.player_step(
                state._replace(player_x=player_x, player_y=player_y), action_mod
            )
            player_missile_position = self.player_missile_step(
                state, next_x, next_y, action_mod
            )

            # Rest of oxygen handling and game logic
            (
                new_oxygen,
                player_x,
                player_y,
                player_missile_position,
                oxygen_depleted,
                lose_life_surfacing,
                new_divers_collected,
                should_reset,
                new_just_surfaced,
                new_difficulty,
            ) = self.update_oxygen(state, next_x, next_y, player_missile_position)

            # Update divers collected count from oxygen mechanics
            state_updated = state_updated._replace(
                divers_collected=new_divers_collected
            )

            # update the spawn state with the new difficulty
            new_spawn_state = state_updated.spawn_state._replace(
                difficulty=new_difficulty
            )

            # Check missile collisions
            (
                player_missile_position,
                new_shark_positions,
                new_sub_positions,
                new_score,
                updated_spawn_state,
                new_rng_key,
            ) = self.check_missile_collisions(
                player_missile_position,
                state_updated.shark_positions,
                state_updated.sub_positions,
                state_updated.score,
                state_updated.successful_rescues,
                new_spawn_state,
                state.rng_key,
            )

            # perform all necessary spawn steps
            (
                new_spawn_state,
                new_shark_positions,
                new_sub_positions,
                new_diver_positions,
                new_rng_key,
            ) = self.spawn_step(
                state_updated,
                updated_spawn_state,
                new_shark_positions,
                new_sub_positions,
                state.diver_positions,
                new_rng_key,
            )

            new_diver_positions, new_divers_collected, new_spawn_state, new_rng_key = (
                self.step_diver_movement(
                    new_diver_positions,
                    new_shark_positions,
                    player_x,
                    player_y,
                    state_updated.divers_collected,
                    new_spawn_state,
                    state_updated.step_counter,
                    new_rng_key,
                )
            )

            new_surface_sub_pos = self.surface_sub_step(state_updated)

            state_updated._replace(surface_sub_position=new_surface_sub_pos)

            # update the enemy missile positions
            new_enemy_missile_positions = self.enemy_missiles_step(
                new_sub_positions,
                state_updated.enemy_missile_positions,
                state_updated.step_counter,
                state_updated.spawn_state.difficulty,
            )

            # append the surface submarine to the other submarines for the collision check
            # check if the player has collided with any of the enemies
            player_collision, collision_points = self.check_player_collision(
                player_x,
                player_y,
                new_sub_positions,
                new_shark_positions,
                new_surface_sub_pos,
                state_updated.enemy_missile_positions,
                new_score,
                state_updated.successful_rescues,
            )

            lose_life = jnp.any(
                jnp.array([oxygen_depleted, player_collision, lose_life_surfacing])
            )

            # Start death animation but keep divers intact during animation
            death_animation_state = state_updated._replace(
                score=state.score + collision_points,
                death_counter=jnp.array(90),
                spawn_state=self.soft_reset_spawn_state(state_updated.spawn_state),
            )

            # Calculate points for rescuing divers. Each diver is worth 50 points.
            # Each successful rescue adds 50 points with a maximum of 1000 points each.
            base_points_per_diver = 50
            max_points_per_diver = 1000
            additional_points_per_rescue = 50 * state.successful_rescues
            points_per_diver = jnp.minimum(
                base_points_per_diver + additional_points_per_rescue,
                max_points_per_diver,
            )
            total_diver_points = points_per_diver * state.divers_collected

            # Calculate bonus points for remaining oxygen
            oxygen_bonus = state.oxygen * 20

            # Calculate total points for successful rescue
            total_rescue_points = total_diver_points + oxygen_bonus

            # TODO: somewhere the oxygen is depleted on surfacing, this currently blocks the slow draining of oxygen (which is not gameplay relevant -> low priority)
            # scoring freeze, 16 ticks per diver i.e. 6 * 16 and also 2 ticks per remaining oxygen (which is drained!)
            # Create the scoring state
            scoring_state = state_updated._replace(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                lives=state_updated.lives,
                score=state_updated.score + total_rescue_points,
                successful_rescues=state_updated.successful_rescues + 1,
                spawn_state=self.soft_reset_spawn_state(state_updated.spawn_state)._replace(
                    difficulty=state_updated.spawn_state.difficulty + 1,
                    survived=state_updated.spawn_state.survived.astype(jnp.int32)
                ),
                death_counter=jnp.array(-(96 + state_updated.oxygen * 2)),
            )

            # cap the step counter to 1024
            new_step_counter = jnp.where(
                state_updated.step_counter == 1024,
                jnp.array(0),
                state_updated.step_counter + 1,
            )

            # Create the normal returned state
            normal_returned_state = SeaquestState(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                oxygen=new_oxygen,
                divers_collected=new_divers_collected,
                score=new_score,
                lives=state_updated.lives,
                spawn_state=new_spawn_state._replace(
                    survived=new_spawn_state.survived.astype(jnp.int32)
                ),
                diver_positions=new_diver_positions,
                shark_positions=new_shark_positions,
                sub_positions=new_sub_positions,
                enemy_missile_positions=new_enemy_missile_positions,
                surface_sub_position=new_surface_sub_pos,
                player_missile_position=player_missile_position,
                step_counter=new_step_counter,
                just_surfaced=new_just_surfaced,
                successful_rescues=state_updated.successful_rescues,
                death_counter=jnp.array(0),
                rng_key=new_rng_key,
            )

            # First handle surfacing with all divers (scoring)
            intermediate_state = jax.lax.cond(
                should_reset,
                lambda _: scoring_state,
                lambda _: normal_returned_state,
                operand=None,
            )

            # Then handle life loss - start death animation instead of immediate reset
            final_state = jax.lax.cond(
                lose_life,
                lambda _: death_animation_state,
                lambda _: intermediate_state,
                operand=None,
            )

            # Check for additional life every 10,000 points
            additional_lives = (final_state.score // 10000) - (state.score // 10000)
            new_lives = jnp.minimum(final_state.lives + additional_lives, 6) # max 6 lives possible

            # Update the final state with new lives
            final_state = final_state._replace(lives=new_lives)

            # Check if the game is over
            game_over = final_state.lives <= -1

            # Handle game over state
            return jax.lax.cond(
                game_over,
                lambda _: state._replace(
                    score=final_state.score,
                    lives=jnp.array(-1),
                    death_counter=jnp.array(0),
                ),
                lambda _: final_state,
                operand=None,
            )

        return_state = jax.lax.cond(
            state.death_counter > 0,
            lambda _: handle_death_animation(),
            lambda _: jax.lax.cond(
                state.death_counter < 0,
                lambda _: handle_score_freeze(),
                lambda _: normal_game_step(),
                operand=None,
            ),
            operand=None,
        )

        # Get observation and info
        observation = self._get_observation(return_state)

        done = self._get_done(return_state)
        env_reward = self._get_env_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

        # Choose between death animation and normal game step
        return observation, return_state, env_reward, done, info


class SeaquestRenderer(JAXGameRenderer):
    def __init__(self, consts: SeaquestConstants = None):
        super().__init__()
        self.offset_length = len(PL_SUB_OFFSETS)
        self.diver_offset_length = len(DIVER_OFFSETS)
        self.shark_offset_length = len(SHARK_OFFSETS)
        self.enemy_sub_offset_length = len(ENEMY_SUB_OFFSETS)
        self.consts = consts or SeaquestConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        # render background
        frame_bg = jr.get_sprite_frame(SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        # render player submarine
        frame_pl_sub = jr.get_sprite_frame(SPRITE_PL_SUB, state.step_counter)
        idx_pl_sub = state.step_counter % self.offset_length
        pl_sub_offset = jnp.take(PL_SUB_OFFSETS, idx_pl_sub, axis=0)
        raster = jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_pl_sub,
            flip_horizontal=state.player_direction == self.consts.FACE_LEFT,
            flip_offset=pl_sub_offset,
        )
        # Player torpedo
        frame_pl_torp = jr.get_sprite_frame(SPRITE_PL_TORP, state.step_counter)
        should_render = state.player_missile_position[0] > 0
        raster = jax.lax.cond(
            should_render,
            lambda r: jr.render_at(
                r,
                state.player_missile_position[0],
                state.player_missile_position[1],
                frame_pl_torp,
                flip_horizontal=state.player_missile_position[2] == self.consts.FACE_LEFT,
            ),
            lambda r: r,
            raster,
        )

        # render divers
        frame_diver = jr.get_sprite_frame(SPRITE_DIVER, state.step_counter)
        diver_positions = state.diver_positions

        def render_diver(i, raster_base):
            should_render = diver_positions[i][0] > 0
            idx_diver = i % self.diver_offset_length
            diver_offset = jnp.take(DIVER_OFFSETS, idx_diver, axis=0)
            return jax.lax.cond(
                should_render,
                lambda r: jr.render_at(
                    r,
                    diver_positions[i][0],
                    diver_positions[i][1],
                    frame_diver,
                    flip_horizontal=(diver_positions[i][2] == self.consts.FACE_LEFT),
                    flip_offset=diver_offset,
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_DIVERS, render_diver, raster)

        # render sharks
        frame_shark = get_shark_sprite_by_difficulty(state.spawn_state.difficulty, state.step_counter)

        def render_shark(i, raster_base):
            should_render = state.shark_positions[i][0] > 0
            idx_shark = i % self.shark_offset_length
            shark_offset = jnp.take(SHARK_OFFSETS, idx_shark, axis=0)
            return jax.lax.cond(
                should_render,
                lambda r: jr.render_at(
                    r,
                    state.shark_positions[i][0],
                    state.shark_positions[i][1],
                    frame_shark,
                    flip_horizontal=(state.shark_positions[i][2] == self.consts.FACE_LEFT),
                    flip_offset=shark_offset,
                ),
                lambda r: r,
                raster_base,
            )

        # Use fori_loop to render all sharks
        raster = jax.lax.fori_loop(0, self.consts.MAX_SHARKS, render_shark, raster)

        # render enemy subs
        frame_enemy_sub = jr.get_sprite_frame(SPRITE_ENEMY_SUB, state.step_counter)

        def render_enemy_sub(i, raster_base):
            should_render = state.sub_positions[i][0] > 0
            idx_enemy_sub = i % self.enemy_sub_offset_length
            enemy_sub_offset = jnp.take(ENEMY_SUB_OFFSETS, idx_enemy_sub, axis=0)
            return jax.lax.cond(
                should_render,
                lambda r: jr.render_at(
                    r,
                    state.sub_positions[i][0],
                    state.sub_positions[i][1],
                    frame_enemy_sub,
                    flip_horizontal=(state.sub_positions[i][2] == self.consts.FACE_LEFT),
                    flip_offset=enemy_sub_offset,
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_SUBS, render_enemy_sub, raster)

        def render_enemy_surface_sub(i, raster_base):
            should_render = state.surface_sub_position[0] > 0
            idx_enemy_sub = i % self.enemy_sub_offset_length
            enemy_sub_offset = jnp.take(ENEMY_SUB_OFFSETS, idx_enemy_sub, axis=0)
            return jax.lax.cond(
                should_render,
                lambda r: jr.render_at(
                    r,
                    state.surface_sub_position[0],
                    state.surface_sub_position[1],
                    frame_enemy_sub,
                    flip_horizontal=(state.surface_sub_position[2] == self.consts.FACE_LEFT),
                    flip_offset=enemy_sub_offset,
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(
            0, self.consts.MAX_SURFACE_SUBS, render_enemy_surface_sub, raster
        )

        frame_enemy_torp = jr.get_sprite_frame(SPRITE_EN_TORP, state.step_counter)

        def render_enemy_torp(i, raster_base):
            should_render = state.enemy_missile_positions[i][0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: jr.render_at(
                    r,
                    state.enemy_missile_positions[i][0],
                    state.enemy_missile_positions[i][1],
                    frame_enemy_torp,
                    flip_horizontal=(state.enemy_missile_positions[i][2] == self.consts.FACE_LEFT),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_MISSILES, render_enemy_torp, raster)

        # show the scores
        score_array = jr.int_to_digits(state.score, max_digits=8)
        # convert the score to a list of digits
        raster = jr.render_label(raster, 10, 10, score_array, DIGITS, spacing=7)
        raster = jr.render_indicator(
            raster, 10, 20, state.lives, LIFE_INDICATOR, spacing=10
        )
        raster = jr.render_indicator(
            raster, 49, 178, state.divers_collected, DIVER_INDICATOR, spacing=10
        )

        raster = jr.render_bar(
            raster, 49, 170, state.oxygen, 64, 63, 5, self.consts.OXYGEN_BAR_COLOR, (0, 0, 0, 0)
        )

        # Force the first 8 columns (x=0 to x=7) to be black
        bar_width = 8
        # Assuming raster shape is (Height, Width, Channels)
        # Select all rows (:), the first 'bar_width' columns (0:bar_width), and all channels (:)
        raster = raster.at[:, :bar_width, :].set(0)

        return raster
