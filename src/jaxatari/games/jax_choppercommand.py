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
import jaxatari.rendering.jax_rendering_utils as render_utils
import numpy as np
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvState
from jaxatari.renderers import JAXGameRenderer
import time

class ChopperCommandConstants:
    # Game Constants
    WINDOW_WIDTH = 160 * 3
    WINDOW_HEIGHT = 210 * 3
    DEATH_PAUSE_FRAMES = 60

    WIDTH = 160
    HEIGHT = 210
    HEIGHT_ONLY_PLAYING_FIELD = 192
    SCALING_FACTOR = 4

    # difficulty
    GAME_DIFFICULTY = 2

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
    SCORE_PER_TRUCK_ALIVE = 100

    # Player Missile Constants
    PLAYER_MISSILE_WIDTH = 80 # Sprite size_x
    MISSILE_COOLDOWN_FRAMES = 8  # DEFAULT: 8 | How fast Chopper can shoot (higher is slower) TODO: Das müssen wir ändern und höher machen bei dem schweren Schwierigkeitsgrad
    MISSILE_SPEED = 10 # DEFAULT: 10 | Missile speed (higher is faster) TODO: tweak MISSILE_SPEED and MISSILE_COOLDOWN_FRAMES to match real game (already almost perfect)
    MISSILE_ANIMATION_SPEED = 6 # DEFAULT: 6 | Rate at which missile changes sprite textures (based on traveled distance of missile)

    # Enemy Missile Constants
    ENEMY_MISSILE_SPAWN_PROBABILITY = 0.01 # The probability that an enemy missiles spawns at one of the living enemies in each frame, if the missile of the giving enemy is not "alive". (Meaning that for 0.01 for example, an enemy shoots a missile on average 100 frames after its previous missile died).
    ENEMY_MISSILE_SPLIT_PROBABILITY = 0.05 # The probability that an enemy missiles splits in a frame
    ENEMY_MISSILE_MAXIMUM_Y_SPEED_BEFORE_SPLIT = 0.75 # Maximum speed (+ and -) of a missile before split. This means that for 2 for example, the missiles will have speeds between -2 and 2 (chosen randomly).
    ENEMY_MISSILE_Y_SPEED_AFTER_SPLIT = 2.5 # TODO: Make match real game

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
    MAX_TRUCKS = 12 # DEFAULT: 12 | How much trucks are spawned
    MAX_JETS = 12 # DEFAULT: 12 | the maximum amount of jets that can be spawned
    MAX_CHOPPERS = 12 # DEFAULT: 12 | the maximum amount of choppers that can be spawned
    MAX_ENEMIES = 12 # DEFAULT: 12 | the amount of enemies that are spawned
    MAX_PLAYER_MISSILES = 1 # DEFAULT: 1 | the original game allows only one missile per screen. The player_missile_step logic is adjusted automatically, if this is changed to more than 1.
    MAX_ENEMY_MISSILES = MAX_ENEMIES * 2 * 2 # Two missiles for every enemy (jets and choppers) (this does not mean, that there are always this many missiles on the screen/in the game)
    ENEMY_LANE_SWITCH_PROBABILITY = 0.05 # DEFAULT: 7 | how likely is it that en enemy switches a lane

    # Enemy movement
    JET_VELOCITY_LEFT = 1.5 # DEFAULT: 1.5 | How fast jets fly to the left
    JET_VELOCITY_RIGHT = 1 # DEFAULT: 1 | How fast jets fly to the right
    CHOPPER_VELOCITY_LEFT = 0.75 # DEFAULT: 0.75 | How fast choppers fly to the right
    CHOPPER_VELOCITY_RIGHT = 0.5 # DEFAULT: 0.5 | How fast choppers fly to the right
    ENEMY_OUT_OF_CYCLE_RIGHT = 64 # DEFAULT: 64 | How far enemies can fly around the truck fleet to the right
    ENEMY_OUT_OF_CYCLE_LEFT = 64 # DEFAULT: 64 | How far enemies cam fly around the truck fleet to the left
    ENEMY_MAXIMUM_SPAWN_OFFSET = 64 # DEFAULT: 64 | How far to the left and right of the middle truck enemies can spawn

    # Enemy Lanes

    ENEMY_LANE_OFFSET = 7 # DEFAULT: 7 | How much apart bottom and top lanes are from the middle lane

    ENEMY_LANE_7 = 66
    ENEMY_LANE_8 = ENEMY_LANE_7 - ENEMY_LANE_OFFSET
    ENEMY_LANE_6 = ENEMY_LANE_7 + ENEMY_LANE_OFFSET

    ENEMY_LANE_4 = 96
    ENEMY_LANE_5 = ENEMY_LANE_4 - ENEMY_LANE_OFFSET
    ENEMY_LANE_3 = ENEMY_LANE_4 + ENEMY_LANE_OFFSET

    ENEMY_LANE_1 = 126
    ENEMY_LANE_2 = ENEMY_LANE_1 - ENEMY_LANE_OFFSET
    ENEMY_LANE_0 = ENEMY_LANE_1 + ENEMY_LANE_OFFSET

    BOTTOM_LANES = jnp.array([ENEMY_LANE_0, ENEMY_LANE_1, ENEMY_LANE_2])
    MIDDLE_LANES = jnp.array([ENEMY_LANE_3, ENEMY_LANE_4, ENEMY_LANE_5])
    TOP_LANES = jnp.array([ENEMY_LANE_6, ENEMY_LANE_7, ENEMY_LANE_8])

    ALL_LANES = jnp.array([ENEMY_LANE_0, ENEMY_LANE_1, ENEMY_LANE_2, ENEMY_LANE_3, ENEMY_LANE_4, ENEMY_LANE_5, ENEMY_LANE_6, ENEMY_LANE_7, ENEMY_LANE_8])

    """
    Correct arrangement of lanes by height is:
    
    ENEMY_LANE_8
    ENEMY_LANE_7
    ENEMY_LANE_6
    
    ENEMY_LANE_5
    ENEMY_LANE_4
    ENEMY_LANE_3
    
    ENEMY_LANE_2
    ENEMY_LANE_1
    ENEMY_LANE_0
    
    """

    # Minimap
    MINIMAP_WIDTH = 48
    MINIMAP_HEIGHT = 16

    MINIMAP_POSITION_X = (WIDTH // 2) - (MINIMAP_WIDTH // 2) # TODO: Im echten Game wird die Minimap nicht mittig, sondern weiter links gerendert. Wir müssen besprechen ob wir das auch machen, dann müsste man nur diese Zahl hier ändern (finde es aber so schöner)
    MINIMAP_POSITION_Y = 165

    MINIMAP_RENDER_TRUCK_REFRESH_RATE = 8 # Higher is slower (Does not fully work yet)

    DOWNSCALING_FACTOR_WIDTH = WIDTH // MINIMAP_WIDTH
    DOWNSCALING_FACTOR_HEIGHT = HEIGHT_ONLY_PLAYING_FIELD // MINIMAP_HEIGHT

    #Object rendering
    TRUCK_SPAWN_DISTANCE = 248 # distance 240px + truck width

    FRAMES_DEATH_ANIMATION_ENEMY = 16
    FRAMES_DEATH_ANIMATION_TRUCK = 32 # TODO: Make match real game
    TRUCK_FLICKER_RATE = 3 # TODO: Make match real game

    PLAYER_FADE_OUT_START_THRESHOLD_0 = 0.25
    PLAYER_FADE_OUT_START_THRESHOLD_1 = 0.125

    # define object orientations
    FACE_LEFT = -1
    FACE_RIGHT = 1

    SPAWN_POSITIONS_Y = jnp.array([60, 90, 120])
    TRUCK_SPAWN_POSITIONS = 156

    # Debugging

    ENABLE_PLAYER_COLLISION = True
    ENABLE_ENEMY_MISSILE_TRUCK_COLLISION = True


class ChopperCommandState(NamedTuple):
    player_x: chex.Array                    # x-coordinate of the player’s chopper in world space
    player_y: chex.Array                    # y-coordinate of the player’s chopper in world space
    player_velocity_x: chex.Array           # horizontal velocity (momentum) of the player’s chopper; positive = moving right, negative = moving left
    local_player_offset: chex.Array         # offset of the player’s chopper from the screen center (used for scrolling logic and chopper’s on-screen position)
    player_facing_direction: chex.Array     # current facing direction of the player’s chopper: -1 = facing left, +1 = facing right, 0 = invalid
    score: chex.Array                       # current game score/points
    lives: chex.Array                       # number of remaining lives for the player
    save_lives: chex.Array                  # keeps track of how often the player was granted a life for reaching another 10000 score points
    truck_positions: chex.Array             # shape (MAX_TRUCKS, 4): for each truck, stores [x, y, direction (active flag), death_timer]
    jet_positions: chex.Array               # shape (MAX_JETS, 4): for each enemy jet, stores [x, y, direction (active flag), death_timer]
    chopper_positions: chex.Array           # shape (MAX_ENEMIES, 4): for each enemy chopper, stores [x, y, direction (active flag), death_timer]
    enemy_missile_positions: chex.Array     # shape (MAX_MISSILES, 4): for each enemy missile, stores [x, y, direction, did_split_flag]
    player_missile_positions: chex.Array    # shape (MAX_MISSILES, 4): for each player missile, stores [x, y, direction, x_coordinate of spawn point]
    player_missile_cooldown: chex.Array     # cooldown timer until the player can fire the next missile
    player_collision: chex.Array            # boolean flag indicating whether the player has collided this frame
    step_counter: chex.Array                # total number of game ticks/frames elapsed so far
    pause_timer: chex.Array                 # counter for how many frames remain in the game pause before respawning; 0 = fully dead, respawn initiated, 1 = either no lives left, infinite pause or ->, 1 - DEATH_PAUSE_FRAMES: counting down for the duration of pause, DEATH_PAUSE_FRAMES + 1 = death_pause, DEATH_PAUSE_FRAMES + 2 = no_move_pause
    rng_key: chex.PRNGKey                   # current PRNG key for any stochastic operations (e.g., random enemy spawns)
    difficulty: chex.Array                  # states the difficulty which can be either 1 or 2
    enemy_speed: chex.Array                 # states the speed of the enemies e.g. all enemies are killed

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

class ChopperCommandObservation(NamedTuple):
    player: PlayerEntity
    trucks: jnp.ndarray # Shape (MAX_TRUCKS, 5) - MAX_TRUCKS enemies, each with x,y,w,h,active
    jets: jnp.ndarray  # Shape (MAX_JETS, 5) - MAX_JETS enemies, each with x,y,w,h,active
    choppers: jnp.ndarray # Shape (MAX_CHOPPERS, 5) - MAX_CHOPPERS enemies, each with x,y,w,h,active
    enemy_missiles: jnp.ndarray  # Shape (MAX_MISSILES, 5)
    player_missile: EntityPosition
    player_score: jnp.ndarray
    lives: jnp.ndarray

class ChopperCommandInfo(NamedTuple):
    step_counter: jnp.ndarray  # Current step count


class JaxChopperCommand(JaxEnvironment[ChopperCommandState, ChopperCommandObservation, ChopperCommandInfo, ChopperCommandConstants]):
    def __init__(self, consts: ChopperCommandConstants = None, reward_funcs: list[callable] = None):
        consts = consts or ChopperCommandConstants()
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

        self.obs_size = (
            6  # player: x,y,o,width,height,active
            + self.consts.MAX_TRUCKS   * 5
            + self.consts.MAX_JETS     * 5
            + self.consts.MAX_CHOPPERS * 5
            + self.consts.MAX_ENEMY_MISSILES * 5
            + 5  # player_missile
            + 1  # player_score
            + 1  # lives
        )

        self.renderer = ChopperCommandRenderer(self.consts)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: ChopperCommandState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.array(
            [entity.x, entity.y, entity.width, entity.height, entity.active],
            dtype=jnp.float32
        )

    def flatten_player_entity(self, entity: PlayerEntity) -> jnp.ndarray:
        return jnp.array(
            [entity.x, entity.y, entity.o, entity.width, entity.height, entity.active],
            dtype=jnp.float32
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: ChopperCommandObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_entity(obs.player),  # 6
            obs.trucks.flatten().astype(jnp.float32),  # 12*5
            obs.jets.flatten().astype(jnp.float32),  # 12*5
            obs.choppers.flatten().astype(jnp.float32),  # 12*5
            obs.enemy_missiles.flatten().astype(jnp.float32),  # 48*5  (alle Missiles)
            self.flatten_entity_position(obs.player_missile),  # 5
            jnp.array([obs.player_score], dtype=jnp.float32),  # 1
            jnp.array([obs.lives], dtype=jnp.float32),  # 1
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        f32 = jnp.float32
        SAFE_X_LOW = -60000.0
        SAFE_X_HIGH = 60000.0

        tw, th = self.consts.TRUCK_SIZE
        jw, jh = self.consts.JET_SIZE
        cw, ch = self.consts.CHOPPER_SIZE
        mw, mh = self.consts.ENEMY_MISSILE_SIZE

        def make_array_box(n, w, h):
            low = jnp.tile(jnp.array([SAFE_X_LOW, 0.0, 0.0, 0.0, 0.0], dtype=f32), (n, 1))
            high = jnp.tile(jnp.array([SAFE_X_HIGH, 210.0, float(w), float(h), 1.0], dtype=f32), (n, 1))
            return spaces.Box(low=low, high=high, dtype=f32)

        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=SAFE_X_LOW, high=SAFE_X_HIGH, shape=(), dtype=f32),
                "y": spaces.Box(low=0.0, high=210.0, shape=(), dtype=f32),
                "o": spaces.Box(low=-1.0, high=1.0, shape=(), dtype=f32),
                "width": spaces.Box(low=0.0, high=160.0, shape=(), dtype=f32),
                "height": spaces.Box(low=0.0, high=210.0, shape=(), dtype=f32),
                "active": spaces.Box(low=0.0, high=1.0, shape=(), dtype=f32),
            }),
            "trucks": make_array_box(self.consts.MAX_TRUCKS, tw, th),
            "jets": make_array_box(self.consts.MAX_JETS, jw, jh),
            "choppers": make_array_box(self.consts.MAX_CHOPPERS, cw, ch),
            "enemy_missiles": make_array_box(self.consts.MAX_ENEMY_MISSILES, mw, mh),

            "player_missile": spaces.Dict({
                "x": spaces.Box(low=SAFE_X_LOW, high=SAFE_X_HIGH, shape=(), dtype=f32),
                "y": spaces.Box(low=0.0, high=210.0, shape=(), dtype=f32),
                "width": spaces.Box(low=0.0, high=160.0, shape=(), dtype=f32),
                "height": spaces.Box(low=0.0, high=210.0, shape=(), dtype=f32),
                "active": spaces.Box(low=0.0, high=1.0, shape=(), dtype=f32),
            }),
            "player_score": spaces.Box(low=0.0, high=999999.0, shape=(), dtype=f32),
            "lives": spaces.Box(low=0.0, high=3.0, shape=(), dtype=f32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for ChopperCommand.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: ChopperCommandState) -> ChopperCommandObservation:
        f32 = jnp.float32

        player = PlayerEntity(
            x=jnp.asarray(state.player_x, dtype=f32),
            y=jnp.asarray(state.player_y, dtype=f32),
            o=jnp.asarray(state.player_facing_direction, dtype=f32),
            width=jnp.asarray(self.consts.PLAYER_SIZE[0], dtype=f32),
            height=jnp.asarray(self.consts.PLAYER_SIZE[1], dtype=f32),
            active=jnp.asarray(1.0, dtype=f32),
        )

        def convert_to_entity(pos, size_wh):
            # aktiv: dir != 0 und death_timer == 0 (oder >FRAMES… – je nach deiner Logik)
            active = jnp.where((pos[2] != 0) & (pos[3] == 0), 1.0, 0.0).astype(f32)
            return jnp.array([pos[0], pos[1], size_wh[0], size_wh[1], active], dtype=f32)

        def convert_missile(pos, size_wh):
            active = jnp.where(pos[2] != 0, 1.0, 0.0).astype(f32)
            return jnp.array([pos[0], pos[1], size_wh[0], size_wh[1], active], dtype=f32)

        trucks = jax.vmap(lambda p: convert_to_entity(p, self.consts.TRUCK_SIZE))(state.truck_positions.astype(f32))
        jets = jax.vmap(lambda p: convert_to_entity(p, self.consts.JET_SIZE))(state.jet_positions.astype(f32))
        choppers = jax.vmap(lambda p: convert_to_entity(p, self.consts.CHOPPER_SIZE))(
            state.chopper_positions.astype(f32))
        enemy_missiles = jax.vmap(lambda p: convert_missile(p, self.consts.ENEMY_MISSILE_SIZE))(
            state.enemy_missile_positions.astype(f32))

        missile_pos = state.player_missile_positions[0].astype(f32)
        player_missile = EntityPosition(
            x=missile_pos[0],
            y=missile_pos[1],
            width=jnp.asarray(self.consts.PLAYER_MISSILE_SIZE[0], dtype=f32),
            height=jnp.asarray(self.consts.PLAYER_MISSILE_SIZE[1], dtype=f32),
            active=jnp.where(missile_pos[2] != 0, 1.0, 0.0).astype(f32),
        )

        player_score = jnp.asarray(state.score, dtype=f32)
        lives = jnp.asarray(state.lives, dtype=f32)

        return ChopperCommandObservation(
            player=player,
            trucks=trucks,
            jets=jets,
            choppers=choppers,
            enemy_missiles=enemy_missiles,  # (MAX_ENEMY_MISSILES,5)
            player_missile=player_missile,
            player_score=player_score,
            lives=lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: ChopperCommandState) -> ChopperCommandInfo:
        return ChopperCommandInfo(
            step_counter=state.step_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: ChopperCommandState, state: ChopperCommandState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: ChopperCommandState) -> bool:
        return state.lives <= 0

    @partial(jax.jit, static_argnums=(0,))
    def check_collision_single(
            self, pos1, size1, pos2, size2
    ):
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
    def check_collision_batch(
            self, pos1, size1, pos2_array, size2
    ):
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
            self,
            enemy_pos: chex.Array,
            death_timer: int
            ) -> chex.Array:
        return jnp.array([
            enemy_pos[0],  # x
            enemy_pos[1],  # y
            enemy_pos[2],  # direction
            death_timer  # death_timer
        ], dtype=enemy_pos.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def check_missile_collisions( # TODO: improve
        self,
        missile_positions: chex.Array,  # (MAX_MISSILES, 4)
        enemy_positions: chex.Array,    # (N_ENEMIES, 2)
        on_screen_position: chex.Array,
        player_x: chex.Array,
        enemy_size: tuple[int, int],
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Check for collisions between player missiles and enemies, mit dynamischer Breitenanpassung."""

        def check_single_missile(missile):
            missile_x, missile_y, direction, _ = missile
            missile_active = missile[3] != 0

            def check_single_enemy(enemy_pos):
                enemy_active = enemy_pos[3] > self.consts.FRAMES_DEATH_ANIMATION_ENEMY

                # Sichtfeldgrenzen
                left_bound = player_x - on_screen_position
                right_bound = left_bound + self.consts.WIDTH

                new_missile = direction

                missile_left = missile_x
                missile_right = missile_x + self.consts.PLAYER_MISSILE_SIZE[0]

                # Dynamische Breite berechnen
                clipped_left = jnp.maximum(missile_left, left_bound)
                clipped_right = jnp.minimum(missile_right, right_bound)

                # Neue Breite
                clipped_width = jnp.maximum(0, clipped_right - clipped_left)

                # Missile ist zu klein → keine Kollision möglich
                too_small = clipped_width <= 0

                # Neue Position für Kollisionstest
                adjusted_pos = jnp.array([clipped_left, missile_y])
                adjusted_size = jnp.array([clipped_width, self.consts.PLAYER_MISSILE_SIZE[1]])

                collision = jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_and(
                            missile_active,
                            enemy_active
                        ),
                        jnp.logical_not(too_small)
                    ),
                    self.check_collision_single(adjusted_pos, adjusted_size, enemy_pos, enemy_size)
                )

                # Kill initialisieren (nicht endgültig tot)
                new_enemy_pos = jnp.where(collision,
                                          self.kill_entity(enemy_pos, self.consts.FRAMES_DEATH_ANIMATION_ENEMY),
                                          enemy_pos)

                # Missile deaktivieren bei Treffer
                new_missile = jnp.where(
                    collision,
                    jnp.array([0, 0, 0, 0], dtype=missile.dtype),
                    missile
                )

                # Punkte vergeben
                is_jet = enemy_size[1] == 6
                score_add = jnp.where(
                    jnp.logical_and(collision, enemy_pos[3] > self.consts.FRAMES_DEATH_ANIMATION_ENEMY),
                    jnp.where(
                        is_jet,
                        self.consts.SCORE_PER_JET_KILL,
                        self.consts.SCORE_PER_CHOPPER_KILL
                    ),
                    0
                )

                return new_enemy_pos, new_missile, score_add

            new_enemy_positions, new_missiles, score_add = jax.vmap(check_single_enemy)(enemy_positions)
            return new_enemy_positions, new_missiles[0], jnp.sum(score_add)                                 # in case of weird behavior: change the array addressing here

        new_enemy_positions, new_missiles, score_add = jax.vmap(check_single_missile)(missile_positions)
        return new_enemy_positions[0], new_missiles, jnp.sum(score_add)



    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision_entity(
        self,
        player_x: chex.Array,
        player_y: chex.Array,
        player_velocity: chex.Array,
        entity_list: chex.Array,
        entity_size: Tuple[int, int],
        death_threshold: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:

        player_pos = jnp.array([player_x, player_y])
        offset = (self.consts.PLAYER_SIZE[0] // 2 - entity_size[0] // 2) - (player_velocity * self.consts.DISTANCE_WHEN_FLYING)

        def check_single(entity):
            #any_collision_inner, updated_entities = carry

            world_x, world_y = entity[0], entity[1]
            death_timer = entity[3]
            is_active = death_timer > death_threshold

            # Passe die Position an, wie sie auch im Renderer korrigiert wird
            adjusted_entity_pos = jnp.array([world_x + offset, world_y])

            # Prüfe Kollision nur bei aktiven Gegnern
            collision = jnp.logical_and(
                is_active,
                self.check_collision_single(player_pos, self.consts.PLAYER_SIZE, adjusted_entity_pos, entity_size)
            )

            # Markiere getroffenen Gegner
            new_entity = jnp.where(collision, self.kill_entity(entity, death_threshold), entity)

            return new_entity

        updated_entity_list = jax.vmap(check_single)(entity_list)

        return jnp.invert(jnp.array_equal(entity_list, updated_entity_list)), updated_entity_list

    @partial(jax.jit, static_argnums=(0,))
    def check_missile_truck_collisions(
            self,
            truck_positions: chex.Array,  # (MAX_TRUCKS, 4): [x, y, direction, death_timer]
            missile_positions: chex.Array,  # (MAX_MISSILES, 4): [x, y, direction, did_split_flag]
            truck_size: Tuple[int, int],  # (width, height)
            missile_size: Tuple[int, int],  # (width, height)
    ) -> Tuple[chex.Array, chex.Array]:

        def check_single_missile(missile, trucks):
            mx, my, _, _ = missile
            missile_active = my > 2

            def check_single_truck(truck):
                tx, ty, tdir, tdeath = truck
                truck_active = jnp.logical_and(tdir != 0, tdeath > 0)

                collision = (
                        missile_active & truck_active &
                        self.check_collision_single(
                            jnp.array([mx, my]), missile_size,
                            jnp.array([tx, ty]), truck_size
                        )
                )

                updated_truck = jnp.where(
                    collision,
                    jnp.array([tx, ty, tdir, self.consts.FRAMES_DEATH_ANIMATION_TRUCK], dtype=truck.dtype),
                    truck
                )

                return collision, updated_truck

            # Check all trucks against this missile
            collision_flags, updated_trucks = jax.vmap(check_single_truck)(trucks)

            # If any collision occurred, mark missile as dead
            collided = jnp.any(collision_flags)
            updated_missile = jnp.where(
                collided,
                jnp.array([0.0, 0.0, 0.0, 187.0], dtype=missile.dtype),
                missile
            )

            return updated_missile, updated_trucks, collision_flags

        # Apply to all missiles
        updated_results = jax.vmap(
            lambda m: check_single_missile(m, truck_positions)
        )(missile_positions)

        updated_missiles = updated_results[0]  # shape: (MAX_MISSILES, 4)
        updated_truck_list = updated_results[1]  # shape: (MAX_MISSILES, MAX_TRUCKS, 4)
        collision_flags_all = updated_results[2]  # shape: (MAX_MISSILES, MAX_TRUCKS)

        # Combine all truck updates from each missile pass
        def reduce_trucks(i, carry):
            combined = carry
            flags = collision_flags_all[i]
            trucks = updated_truck_list[i]
            return jnp.where(flags[:, None], trucks, combined)

        initial_trucks = truck_positions
        updated_trucks = jax.lax.fori_loop(0, missile_positions.shape[0], reduce_trucks, initial_trucks)

        return updated_trucks, updated_missiles

    @partial(jax.jit, static_argnums=(0,))
    def initialize_enemy_positions(self, init_rng: chex.PRNGKey, ref_x: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Spawn enemy fleets, but wrap each fleet's anchor so it is near ref_x.
        This prevents origin-spawn artifacts during the no-move pause and keeps
        the minimap consistent immediately after respawn.
        """
        jet_positions = jnp.zeros((self.consts.MAX_ENEMIES, 4), dtype=jnp.float32)
        chopper_positions = jnp.zeros((self.consts.MAX_ENEMIES, 4), dtype=jnp.float32)

        # Wrap helper: put x within +-624 of ref
        def wrap_near_ref(x, ref):
            period = jnp.asarray(1248.0, dtype=x.dtype)  # 2 * 624
            half = jnp.asarray(624.0, dtype=x.dtype)
            return ref + jnp.mod(x - ref + half, period) - half

        ref_x = jnp.asarray(ref_x, dtype=jnp.float32)

        # Fleet layout
        fleet_start_x = jnp.asarray(-780.0, dtype=jnp.float32)
        fleet_spacing_x = jnp.asarray(312.0, dtype=jnp.float32)
        fleet_count = 4
        units_per_fleet = 3
        vertical_spacing = 30
        y_start = self.consts.HEIGHT_ONLY_PLAYING_FIELD // 2 - (units_per_fleet // 2) * vertical_spacing

        # RNG
        key0, key1, key2 = jax.random.split(init_rng, 3)
        keys_for_direction = jax.random.split(key0, fleet_count)
        keys_for_chopper_amount = jax.random.split(key1, fleet_count)
        keys_for_offsets = jax.random.split(key2, fleet_count * units_per_fleet)

        carry = (jet_positions, chopper_positions, 0)

        def spawn_fleet(fleet_idx, carry):
            jet_positions, chopper_positions, global_idx = carry

            # Base anchor and wrap it near ref_x
            base_anchor = fleet_start_x + fleet_spacing_x * jnp.asarray(fleet_idx, dtype=jnp.float32)
            anchor_x = wrap_near_ref(base_anchor, ref_x)

            # Random directions per unit
            directions = jax.random.choice(
                keys_for_direction[fleet_idx],
                jnp.array([-1.0, 1.0], dtype=jnp.float32),
                shape=(units_per_fleet,),
                replace=True
            )

            # Random number of choppers in this fleet
            chopper_count = jax.random.randint(keys_for_chopper_amount[fleet_idx], (), 0, units_per_fleet + 1)

            # X offsets per unit
            x_offset_array = jnp.array([
                jax.random.randint(keys_for_offsets[fleet_idx * units_per_fleet + 0], (),
                                   -self.consts.ENEMY_MAXIMUM_SPAWN_OFFSET + 5,
                                   self.consts.ENEMY_MAXIMUM_SPAWN_OFFSET - 5),
                jax.random.randint(keys_for_offsets[fleet_idx * units_per_fleet + 1], (),
                                   -self.consts.ENEMY_MAXIMUM_SPAWN_OFFSET + 5,
                                   self.consts.ENEMY_MAXIMUM_SPAWN_OFFSET - 5),
                jax.random.randint(keys_for_offsets[fleet_idx * units_per_fleet + 2], (),
                                   -self.consts.ENEMY_MAXIMUM_SPAWN_OFFSET + 5,
                                   self.consts.ENEMY_MAXIMUM_SPAWN_OFFSET - 5),
            ], dtype=jnp.float32)

            def place_unit(i, unit_carry):
                jet_positions, chopper_positions, jet_idx, chopper_idx = unit_carry
                y = jnp.asarray(y_start + i * vertical_spacing, dtype=jnp.float32)
                offset_x = x_offset_array[i]
                direction = directions[i]
                pos = jnp.array([anchor_x + offset_x, y, direction, self.consts.FRAMES_DEATH_ANIMATION_ENEMY + 5.0], dtype=jnp.float32)

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

                jet_idx = jet_idx + jnp.where(is_chopper, 0, 1)
                chopper_idx = chopper_idx + jnp.where(is_chopper, 1, 0)
                return jet_positions, chopper_positions, jet_idx, chopper_idx

            jet_positions, chopper_positions, jet_idx, chopper_idx = jax.lax.fori_loop(
                0, units_per_fleet, place_unit, (jet_positions, chopper_positions, global_idx, global_idx)
            )

            new_global_idx = global_idx + units_per_fleet
            return (jet_positions, chopper_positions, new_global_idx)

        jet_positions, chopper_positions, _ = jax.lax.fori_loop(0, fleet_count, spawn_fleet, carry)
        return jet_positions, chopper_positions

    @partial(jax.jit, static_argnums=(0,))
    def emit_enemy_speed(self, speed: chex.Array) -> chex.Array:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def step_enemy_movement(
            self,
            truck_positions: chex.Array,
            jet_positions: chex.Array,
            chopper_positions: chex.Array,
            rng: chex.PRNGKey,
            state_player_x: chex.Array,
            local_player_offset: chex.Array,
            difficulty: chex.Array,
            enemy_speed: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:

        rng0, direction_rng = jax.random.split(rng)
        rng1, process_jet_rng = jax.random.split(rng0)
        return_rng, process_chopper_rng = jax.random.split(rng1)

        def emit_velocity() -> chex.Array:
            difficulty_value = jnp.where(difficulty == 1, 1, 1.5)
            jet_speed_right = (self.consts.JET_VELOCITY_RIGHT + enemy_speed) * difficulty_value
            jet_speed_left = (self.consts.JET_VELOCITY_LEFT + enemy_speed) * difficulty_value
            chopper_speed_right = (self.consts.CHOPPER_VELOCITY_RIGHT + enemy_speed) * difficulty_value
            chopper_speed_left = (self.consts.CHOPPER_VELOCITY_LEFT + enemy_speed) * difficulty_value

            return jnp.array([jet_speed_right, jet_speed_left, chopper_speed_right, chopper_speed_left])


        def move_enemy_x(pos: chex.Array, is_jet, move_x_is_in_range) -> chex.Array:
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

                return jnp.logical_or(enemy_pos[0] > nearest_middle_truck + self.consts.ENEMY_OUT_OF_CYCLE_RIGHT,
                                      enemy_pos[0] < nearest_middle_truck - self.consts.ENEMY_OUT_OF_CYCLE_LEFT)


            def is_in_range_function():
                def check_if_out_of_cycle_jet_function():
                    def jet_is_out_of_cycle_function():

                        return jax.lax.cond( # jet_is_out_of_cycle_function
                            pos[2] == -1,
                            lambda _: jnp.array([pos[0] + pos[2] * -1 * emit_velocity()[1] - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32), # if enemy direction is -1 and position out of bounds turn around by multiply direction with -1 (LEFT -> RIGHT) and adding the LEFT_VELOCITY * direction to the position
                            lambda _: jnp.array([pos[0] + pos[2] * -1 * emit_velocity()[0] - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32), # if enemy direction is not -1 and position out of bounds turn around by multiply direction with -1 (RIGHT -> LEFT) and adding the RIGHT_VELOCITY * direction to the position
                            operand=None
                        )

                    def jet_is_not_out_of_cycle_function():

                        return jax.lax.cond( # jet_is_not_out_of_cycle_function
                            pos[2] == -1,
                            lambda _: jnp.array([pos[0] + pos[2] * emit_velocity()[1], pos[1], pos[2], pos[3]], dtype=jnp.float32), # if enemy direction is -1 and position is not out of bounds just add the LEFT_VELOCITY * direction onto the position
                            lambda _: jnp.array([pos[0] + pos[2] * emit_velocity()[0], pos[1], pos[2], pos[3]], dtype=jnp.float32), # if enemy direction is not -1 and position is not out of bounds just add the RIGHT_VELOCITY * direction onto the position
                            operand=None
                        )


                    return jax.lax.cond( # is_in_range_function
                        is_out_of_cycle(pos),                               # checks if enemy is out of cycle
                        lambda _: jet_is_out_of_cycle_function(),           # Jet is indeed out of bounds
                        lambda _: jet_is_not_out_of_cycle_function(),       # Jet is inside of bounds
                        operand=None
                    )


                def check_if_out_of_cycle_chopper_function():

                    def chopper_is_out_of_cycle_function():

                        return jax.lax.cond( # chopper_is_out_of_cycle_function
                            pos[2] == -1,
                            lambda _: jnp.array([pos[0] + pos[2] * -1 * emit_velocity()[3] - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32),  # Since chopper is out of bounds, change its direction from left to right and do one step to the right + 1 to escape cycle (bounds) boundry
                            lambda _: jnp.array([pos[0] + pos[2] * -1 * emit_velocity()[2] - 1, pos[1], pos[2] * -1, pos[3]], dtype=jnp.float32), # Since chopper is out of bounds, change its direction from right to left and do one step to the left + 1 to escape cycle (bounds) boundry
                            operand=None
                        )

                    def chopper_is_not_out_of_cycle_function():

                        return jax.lax.cond( # chopper_is_not_out_of_cycle_function
                            pos[2] == -1,
                            lambda _: jnp.array([pos[0] + pos[2] * emit_velocity()[3], pos[1], pos[2], pos[3]], dtype=jnp.float32),  # Since chopper is inside of bounds, move chopper to the left by the defined speed
                            lambda _: jnp.array([pos[0] + pos[2] * emit_velocity()[2], pos[1], pos[2], pos[3]], dtype=jnp.float32), # Since chopper is inside of bounds, move chopper to the right by the defined speed
                            operand=None
                        )


                    return jax.lax.cond( # check_if_out_of_cycle_chopper_function
                        is_out_of_cycle(pos),
                        lambda _: chopper_is_out_of_cycle_function(),     # Chopper is indeed out of bounds
                        lambda _: chopper_is_not_out_of_cycle_function(), # Chopper is inside of bounds
                        operand=None
                    )


                return jax.lax.cond( # is_in_range_function
                    is_jet,
                    lambda _: check_if_out_of_cycle_jet_function(),       # Check if jet is out of bounds
                    lambda _: check_if_out_of_cycle_chopper_function(),   # Check if chopper is out of bounds
                    operand=None
                )


            def not_active_or_not_in_range(): # looks for in_range and active enemies from all fleets
                return jax.lax.cond(
                    move_x_is_in_range,
                    lambda _: pos,                                                      # is dead -> does not move at all
                    lambda _: jnp.array([pos[0] - 0.5, pos[1], pos[2], pos[3]]),        # moves the same speed as trucks to the left (fleet + trucks stay freezed)
                    operand=None
                )

            new_pos = jax.lax.cond(
                jnp.logical_and(is_active, move_x_is_in_range),
                lambda _: is_in_range_function(),           # Enemy is alive and is in the nearest fleet to the player
                lambda _: not_active_or_not_in_range(),     # Enemy is either dead or not part of the nearest fleet to the player.
                operand=None
            )


            out_of_bounds = jnp.abs(state_player_x - pos[0]) > 624
            wrapped_x = pos[0] + jnp.clip(state_player_x - pos[0], -1, 1) * 1248  # + pos[2] * 0.5
            wrapped_pos = jnp.array([wrapped_x, pos[1], pos[2], pos[3]])

            return jnp.where(out_of_bounds, wrapped_pos, new_pos)

        def move_enemy_y(pos: chex.Array, move_y_rng_key: chex.Array, middle_lane_rng_key: chex.Array, move_y_is_in_range: chex.Array):
            is_active = pos[2] != 0
            is_on_correct_lane = jnp.logical_or(pos[1] == (pos[3] - self.consts.FRAMES_DEATH_ANIMATION_ENEMY), pos[3] == self.consts.FRAMES_DEATH_ANIMATION_ENEMY + 5) # If enemy is on the lane defined in its lane flag, except if lane flag equals the reset state

            def dont_touch_position():
                return pos[1], pos[3]

            def move_or_dont():
                def dont_move_and_reset_lane_flag():
                    return pos[1], self.consts.FRAMES_DEATH_ANIMATION_ENEMY + 5.0

                def move():
                    def get_go_up():
                        return pos[1] - 0.5, pos[3]
                    def get_go_down():
                        return pos[1] + 0.5, pos[3]

                    lane_is_below = pos[1] < (pos[3] - self.consts.FRAMES_DEATH_ANIMATION_ENEMY)
                    return jax.lax.cond(lane_is_below,
                                        lambda _: get_go_down(),  # If lane is below, do one step towards ground
                                        lambda _: get_go_up(),    # If lane is above, do one step towards sky
                                        operand=None)

                return jax.lax.cond( # move_or_dont
                    is_on_correct_lane,
                    lambda _: dont_move_and_reset_lane_flag(),   # Enemy is on desired lane (allow new lane to be picked)
                    lambda _: move(),                            # Move enemy towards desired lane
                    operand=None
                )

            def get_next_pos():
                def pick_desired_lane():
                    is_in_top_lane = jnp.any(pos[1] == self.consts.TOP_LANES)           # If enemy is in the global top lane
                    is_in_middle_lane = jnp.any(pos[1] == self.consts.MIDDLE_LANES)     # If enemy is in the glocal middle lane

                    new_lane = self.consts.FRAMES_DEATH_ANIMATION_ENEMY + jnp.where( # Pick a valid lane for the enemy. (Pick one local lane from the global lane the enemy is on)
                        is_in_top_lane,
                        jax.random.choice(move_y_rng_key, self.consts.TOP_LANES).astype(jnp.float32),   # RNG FOR LANE PICK IS NOT FULLY RANDOM, LIKE IN REAL GAME. CHANGE ALL OCCURRENCES OF move_y_rng_keys[1] to move_y_rng_keys[i] FOR IT TO BE FULLY RANDOM
                        jnp.where(
                            is_in_middle_lane,
                            jax.random.choice(middle_lane_rng_key, self.consts.MIDDLE_LANES).astype(jnp.float32),
                            jax.random.choice(move_y_rng_key, self.consts.BOTTOM_LANES).astype(jnp.float32)  # If not in top or middle, enemy has to be in the bottom lane
                        )
                    )
                    return pos[1], new_lane


                should_switch = jax.random.bernoulli(middle_lane_rng_key, p=self.consts.ENEMY_LANE_SWITCH_PROBABILITY)
                return jax.lax.cond( # get_next_pos
                    jnp.logical_and(should_switch, is_on_correct_lane),
                    lambda _: pick_desired_lane(),  # Since enemy is on its desired lane and should_switch is triggered, we need to update the lane handler to the desired lane
                    lambda _: move_or_dont(),       # Move enemy if not on desired lane or don't move if on the desired lane
                    operand=None
                )

            (new_y, death_timer_or_lane_flag) = jax.lax.cond( # move_enemy_y
                jnp.logical_and((pos[3] > self.consts.FRAMES_DEATH_ANIMATION_ENEMY + 4), jnp.logical_and(is_active, move_y_is_in_range)),
                lambda _: get_next_pos(),           # If pos[3] is interpreted as the lane handler
                lambda _: dont_touch_position(),    # If pos[3] is interpreted as the death timer, also used for logic where the y position should not be changed
                operand=None
            )

            return jnp.array([pos[0], new_y, pos[2], death_timer_or_lane_flag], dtype=pos.dtype)

        def is_in_range_checker(pos: chex.Array) -> chex.Array:
            all_middle_trucks_x = jnp.array(
                [
                    truck_positions[1][0],
                    truck_positions[4][0],
                    truck_positions[7][0],
                    truck_positions[10][0],
                ]
            )

            # nearest_middle_truck = x-coordinate of middle truck closest to enemy with given pos
            distance_to_enemy = jnp.abs(all_middle_trucks_x - pos[0])
            min_index = jnp.argmin(distance_to_enemy)
            nearest_middle_truck = all_middle_trucks_x[min_index]

            # nearest_middle_truck_player = x-coordinate of middle truck closest to player
            distance_to_player = jnp.abs(all_middle_trucks_x - state_player_x + local_player_offset)
            min_index_player = jnp.argmin(distance_to_player)
            nearest_middle_truck_player = all_middle_trucks_x[min_index_player]

            return jnp.where(nearest_middle_truck == nearest_middle_truck_player, jnp.array(True), jnp.array(False))

        def process_enemy(current_pos, key, middle_lane_key, is_jet):                                                                      # Get current position
            enemy_is_in_range = is_in_range_checker(current_pos)                                                  # If jet is in "render distance". If not, it does not move
            new_pos = move_enemy_x(current_pos, is_jet, enemy_is_in_range)                               # Calculate new x position
            new_pos = move_enemy_y(new_pos, key, middle_lane_key, enemy_is_in_range)                           # Calculate new y position
            return new_pos

        # --Process jets --
        # Prepare rng keys for random jet movement TODO: randomization might need to be fixed
        keys_for_process_jet = jax.random.split(process_jet_rng, jet_positions.shape[0])  # Generate rng-key for every chopper there is. THIS IS NOT FULLY USED, SINCE MOVEMENT IS NOT FULLY RANDOM.
        middle_lane_jet_key = keys_for_process_jet[1]
        #new_jet_positions = jnp.zeros_like(jet_positions)   # for debugging purposes
        new_jet_positions = jax.vmap(
            lambda a, b: process_enemy(a, b, middle_lane_jet_key, jnp.array(True)),
            in_axes=(0, 0)
        )(jet_positions, keys_for_process_jet)

        # -- Process choppers --
        # Prepare rng keys for random jet movement TODO: randomization might need to be fixed
        keys_for_process_chopper = jax.random.split(process_chopper_rng, jet_positions.shape[0])  # Generate rng-key for every chopper there is. THIS IS NOT FULLY USED, SINCE MOVEMENT IS NOT FULLY RANDOM.
        middle_lane_chopper_key = keys_for_process_jet[1]
        #new_chopper_positions = jnp.zeros_like(chopper_positions)  # for debugging purposes
        new_chopper_positions = jax.vmap(
            lambda a, b: process_enemy(a, b, middle_lane_chopper_key, jnp.array(False)),
            in_axes=(0, 0)
        )(chopper_positions, keys_for_process_chopper)

        return new_jet_positions, new_chopper_positions, return_rng

    def update_entity_death(self, entity_array, death_timer, is_truck):
        def update_entity(entity):
            direction, timer = entity[2], entity[3]

            #Wenn Tod initialisiert (also noch aktiv) und timer > 0 & timer <= FRAMES_DEATH_ANIMATION, dann dekrementieren
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

            return new_entity

        return jax.vmap(update_entity)(entity_array)


    @partial(jax.jit, static_argnums=(0,))
    def initialize_truck_positions(self) -> chex.Array:     # TODO: change to vmap (via positioning array) if possible
        initial_truck_positions = jnp.zeros((self.consts.MAX_TRUCKS, 4))
        anchor = -748
        carry = (initial_truck_positions, anchor)

        def spawn_trucks(i, carry):
            truck_positions, anchor = carry

            anchor = jnp.where(
                i % 3 == 0,
                anchor + 248,
                anchor + 32,
            )
            truck_positions = truck_positions.at[i].set(jnp.array([anchor, 156, -1, self.consts.FRAMES_DEATH_ANIMATION_TRUCK + 1]))
            return truck_positions, anchor

        return jax.lax.fori_loop(0, 12, spawn_trucks, carry)[0]

    @partial(jax.jit, static_argnums=(0,))
    def step_truck_movement(
            self,
            truck_positions: chex.Array,
            state_player_x: chex.Array,
    ) -> chex.Array:

        def move_single_truck(truck_pos):
            movement_x = -0.5  # Geschwindigkeit 0.5 pro Frame, egal ob tot oder nicht, weil wir die Position noch für die enemy positions brauchen

            out_of_bounds = jnp.abs(state_player_x - truck_pos[0]) > 624

            new_x = jnp.where(
                out_of_bounds,
                truck_pos[0] + jnp.sign(state_player_x - truck_pos[0]) * 1248 + movement_x,
                truck_pos[0] + movement_x,
            )

            new_pos = jnp.array([new_x, truck_pos[1], truck_pos[2], truck_pos[3]])

            return new_pos

        return jax.vmap(move_single_truck)(truck_positions)


    @partial(jax.jit, static_argnums=(0,))
    def enemy_missiles_step(        # TODO: vmap
            self,
            jet_positions: chex.Array,  # (MAX_ENEMIES, 4)
            chopper_positions: chex.Array,  # (MAX_ENEMIES, 4)
            missile_states: chex.Array,  # (MAX_ENEMY_MISSILES, 4): [x, y, y_dir, did_split]
            rng: chex.PRNGKey,
    ) -> chex.Array:

        enemies = jnp.concatenate([jet_positions, chopper_positions], axis=0)

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
                        minval=-self.consts.ENEMY_MISSILE_MAXIMUM_Y_SPEED_BEFORE_SPLIT,
                        maxval=self.consts.ENEMY_MISSILE_MAXIMUM_Y_SPEED_BEFORE_SPLIT
                    )

                    # We also set the did_split flag to false
                    spawned_upper_missile = jnp.array([x_upper_spawn, y_upper_spawn, y_speed_spawn, 187.0], dtype=jnp.float32)
                    spawned_lower_missile = jnp.array([x_lower_spawn, y_lower_spawn, y_speed_spawn, 187.0], dtype=jnp.float32)

                    return spawned_upper_missile, spawned_lower_missile

                return jax.lax.cond(
                    jax.random.bernoulli(key_spawn, p=self.consts.ENEMY_MISSILE_SPAWN_PROBABILITY),
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
                    y_upper_speed_inner = jnp.array(self.consts.ENEMY_MISSILE_Y_SPEED_AFTER_SPLIT, dtype=jnp.float32)
                    y_lower_speed_inner = jnp.array(-self.consts.ENEMY_MISSILE_Y_SPEED_AFTER_SPLIT, dtype=jnp.float32)
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
                    jax.random.bernoulli(key_split, p=self.consts.ENEMY_MISSILE_SPLIT_PROBABILITY),
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


    @partial(jax.jit, static_argnums=(0,))
    def player_missile_step(
            self,
            state: ChopperCommandState,
            curr_player_x,
            curr_player_y,
            action: chex.Array,
    ):
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

        def try_spawn(missiles): # TODO: rewrite
            def body(i, carry):
                missiles, did_spawn = carry
                missile = missiles[i]
                free = missile[2] == 0  # direction == 0 -> inactive
                should_spawn = jnp.where(self.consts.MAX_PLAYER_MISSILES > 1,
                                         jnp.logical_and(free, jnp.logical_not(did_spawn)),
                                         jnp.array(True))

                spawn_x = jnp.where(
                    state.player_facing_direction == -1,
                    curr_player_x - self.consts.PLAYER_MISSILE_WIDTH,
                    curr_player_x + self.consts.PLAYER_SIZE[0],
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
            return jax.lax.cond(jnp.logical_and(jnp.logical_and(fire, state.pause_timer > self.consts.DEATH_PAUSE_FRAMES), cooldown == 0), do_spawn, skip_spawn, operand=None)

        def update_missile(missile):
            exists = missile[2] != 0
            new_x = missile[0] + missile[2] * self.consts.MISSILE_SPEED + state.player_velocity_x

            updated = jnp.array([
                new_x,        # updated x
                missile[1],   # y stays
                missile[2],   # direction stays
                missile[3]    # x_spawn stays
            ], dtype=jnp.int32)

            chopper_pos = (self.consts.WIDTH // 2) - 8 + state.local_player_offset + (state.player_velocity_x * self.consts.DISTANCE_WHEN_FLYING)
            left_bound = state.player_x - chopper_pos - self.consts.PLAYER_MISSILE_WIDTH
            right_bound = state.player_x + (self.consts.WIDTH - chopper_pos)

            out_of_bounds = jnp.logical_or(updated[0] < left_bound, updated[0] > right_bound)
            return jnp.where(jnp.logical_and(exists, ~out_of_bounds), updated, jnp.array([0, 0, 0, 0], dtype=jnp.int32))

        updated_missiles = jax.vmap(update_missile)(state.player_missile_positions)
        # jax.debug.print("{}", updated_missiles)
        updated_missiles, did_spawn = spawn_if_possible(updated_missiles)
        new_cooldown = jnp.where(did_spawn, self.consts.MISSILE_COOLDOWN_FRAMES, cooldown)

        return updated_missiles, new_cooldown


    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state: ChopperCommandState,
        action: chex.Array,
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
        accel_x = jnp.where(right, self.consts.ACCEL, jnp.where(left, -self.consts.ACCEL, 0.0))

        # Direction player is facing
        new_player_facing_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_facing_direction))

        # Neue Geschwindigkeit berechnen und begrenzen
        velocity_x = state.player_velocity_x + accel_x
        velocity_x = jnp.clip(velocity_x, -self.consts.MAX_VELOCITY, self.consts.MAX_VELOCITY)

        # Falls keine Eingabe: langsamer werden (Friction)
        velocity_x = jnp.where(~(left | right), velocity_x * (1.0 - self.consts.FRICTION), velocity_x)

        # Neue X-Position (global!)
        player_x = state.player_x + velocity_x

        # Y-Position berechnen (sofortige Reaktion)
        delta_y = jnp.where(up, -1, jnp.where(down, 1, 0))
        player_y = jnp.clip(state.player_y + delta_y, self.consts.PLAYER_BOUNDS[1][0], self.consts.PLAYER_BOUNDS[1][1])

        # "Momentum" berechnen für Offset von der Mitte aus
        new_player_offset = jnp.where(new_player_facing_direction == 1, state.local_player_offset - self.consts.LOCAL_PLAYER_OFFSET_SPEED, state.local_player_offset + self.consts.LOCAL_PLAYER_OFFSET_SPEED)
        new_player_offset = jnp.asarray(new_player_offset, dtype=jnp.int32)

        new_player_offset = jnp.clip(new_player_offset, -60, 60)

        return player_x, player_y, velocity_x, new_player_offset, new_player_facing_direction

    @partial(jax.jit, static_argnums=(0,))
    def lives_step(
            self,
            player_collision: chex.Array,
            current_score: jnp.int32,
            save_lives: jnp.int32,
    ) -> tuple[jnp.int32, jnp.int32]:

        current_score = jnp.where(current_score == 0, 1, current_score)
        num_to_add = jnp.where(player_collision, -1, 0)

        num_to_add = jnp.where((current_score // 10000) > save_lives, num_to_add + 1, num_to_add)
        new_save_lives = jnp.where((current_score // 10000) > save_lives, save_lives + 1, save_lives)

        return num_to_add, new_save_lives

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(time.time_ns() % (2 ** 32))) -> Tuple[
        ChopperCommandObservation, ChopperCommandState]:
        """Initialize game state"""

        new_key0, key0 = jax.random.split(key, 2)

        # Pass ref_x so fleets spawn wrapped near the initial player position
        jet_positions, chopper_positions = self.initialize_enemy_positions(
            key0, jnp.asarray(self.consts.PLAYER_START_X, dtype=jnp.float32)
        )
        initial_enemy_missile_positions = jnp.full((self.consts.MAX_ENEMY_MISSILES, 4), jnp.array([0, 0, 0, 187], dtype=jnp.float32))

        # FOR EXPLANATION OF FIELDS SEE ChopperCommandState CLASS
        # Explanation for concrete values below
        reset_state = ChopperCommandState(
            player_x=jnp.array(self.consts.PLAYER_START_X).astype(jnp.float32),             # Initial horizontal spawn position of the player in world space. Sets where the chopper starts.
            player_y=jnp.array(self.consts.PLAYER_START_Y).astype(jnp.int32),               # Initial vertical spawn position of the player. Ensures the chopper appears above ground level.
            player_velocity_x=jnp.array(0).astype(jnp.float32),                             # Player starts at rest horizontally — no momentum on the first frame.
            local_player_offset=jnp.array(50).astype(jnp.float32),                          # Initial screen offset. This is for the no_move_pause animation when starting the game and respawning.
            player_facing_direction=jnp.array(1).astype(jnp.int32),                         # # Player begins facing right (1).
            score=jnp.array(0).astype(jnp.int32),                                           # Game always starts with a score of 0.
            lives=jnp.array(3).astype(jnp.int32),                                           # Standard number of starting lives.
            save_lives=jnp.array(0).astype(jnp.int32),                                      # Player starts with no lives granted for every 10000 points reached
            truck_positions=self.initialize_truck_positions().astype(jnp.float32),          # Trucks are initialized with predefined starting positions and inactive death timers.
            jet_positions=jet_positions,                                                    # Jets are initialized with predefined starting positions and inactive death timers.
            chopper_positions=chopper_positions,                                            # Choppers are initialized with predefined starting positions and inactive death timers.
            enemy_missile_positions=initial_enemy_missile_positions,                        # All enemy missile entries are coded as "dead": [0.0, 0.0, 0.0, 187.0], meaning no missiles are in play at start.
            player_missile_positions=jnp.zeros((self.consts.MAX_PLAYER_MISSILES, 4)),       # All player missile slots are zeroed out; meaning no missiles are in play at start.
            player_missile_cooldown=jnp.array(0),                                           # Cooldown timer is 0, so the player is allowed to shoot immediately.
            player_collision=jnp.array(False),                                              # Player has not collided with anything on game start.
            step_counter=jnp.array(0).astype(jnp.int32),                                    # Frame counter starts from 0.
            pause_timer=jnp.array(self.consts.DEATH_PAUSE_FRAMES + 2).astype(jnp.int32),    # The game starts in the no_move_pause (DEATH_PAUSE_FRAMES + 2) to allow for visual startup or intro.
            rng_key=new_key0,                                                               # Pseudo random number generator seed key, based on current time and initial key used.
            difficulty=jnp.array(self.consts.GAME_DIFFICULTY).astype(jnp.float32),          # difficulty of game
            enemy_speed=jnp.array(0).astype(jnp.float32),                                   # enemy_speed which is 0 on start
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state


    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ChopperCommandState, action: chex.Array):
        prev = state  # for reward/info

        def _match_state_dtypes(out_state, ref_state):
            # Casts every jnp.ndarray in out_state to ref_state's dtype (field-wise)
            return jax.tree.map(
                lambda new, old: new.astype(old.dtype) if (hasattr(new, "dtype") and hasattr(old, "dtype")) else new,
                out_state, ref_state
            )

        # -------- Determine phase --------
        in_death_pause   = (state.pause_timer > 0) & (state.pause_timer <= self.consts.DEATH_PAUSE_FRAMES)
        in_no_move_pause = (state.pause_timer == self.consts.DEATH_PAUSE_FRAMES + 2)
        needs_respawn    = (state.pause_timer == 0)

        # 0=NORMAL, 1=DEATH_PAUSE, 2=NO_MOVE, 3=RESPAWN  (explicit int32)
        phase = jnp.where(
            needs_respawn, jnp.int32(3),
            jnp.where(
                in_death_pause, jnp.int32(1),
                jnp.where(in_no_move_pause, jnp.int32(2), jnp.int32(0))
            )
        )

        # -------- Branches --------
        def do_normal(_):
            # RNG split
            k_move, k_mis, new_rng = jax.random.split(state.rng_key, 3)

            # Player
            new_px, new_py, new_vx, new_loc_off, new_face = self.player_step(state, action)

            # Enemies + Trucks
            new_jet, new_chop, new_rng = self.step_enemy_movement(
                state.truck_positions, state.jet_positions, state.chopper_positions,
                k_move, state.player_x, state.local_player_offset,
                state.difficulty, state.enemy_speed
            )
            new_truck = self.step_truck_movement(state.truck_positions, state.player_x)

            # Missiles
            new_enemy_mis = self.enemy_missiles_step(new_jet, new_chop, state.enemy_missile_positions, k_mis)
            new_player_mis, new_cd = self.player_missile_step(state, state.player_x, state.player_y, action)

            # On-screen position
            on_screen = (self.consts.WIDTH // 2) - 8 + new_loc_off + (new_vx * self.consts.DISTANCE_WHEN_FLYING)

            # Player missile vs jets/choppers
            new_jet, new_player_mis, add_jet = self.check_missile_collisions(new_player_mis, new_jet,  on_screen, new_px, self.consts.JET_SIZE)
            new_chop, new_player_mis, add_chop = self.check_missile_collisions(new_player_mis, new_chop, on_screen, new_px, self.consts.CHOPPER_SIZE)

            # Player vs entities/missiles
            pc_jet, new_jet = self.check_player_collision_entity(new_px, new_py, new_vx, new_jet, self.consts.JET_SIZE, self.consts.FRAMES_DEATH_ANIMATION_ENEMY)
            pc_chp, new_chop = self.check_player_collision_entity(new_px, new_py, new_vx, new_chop, self.consts.CHOPPER_SIZE, self.consts.FRAMES_DEATH_ANIMATION_ENEMY)
            pc_trk, new_truck = self.check_player_collision_entity(new_px, new_py, new_vx, new_truck, self.consts.TRUCK_SIZE, self.consts.FRAMES_DEATH_ANIMATION_TRUCK)
            pc_mis, new_enemy_mis = self.check_player_collision_entity(new_px, new_py, new_vx, new_enemy_mis, self.consts.ENEMY_MISSILE_SIZE, 0)

            player_coll = jnp.where(self.consts.ENABLE_PLAYER_COLLISION, (pc_jet | pc_chp | pc_trk | pc_mis), False)

            # Enemy missile vs trucks
            trk2, mis2 = self.check_missile_truck_collisions(new_truck, new_enemy_mis, self.consts.TRUCK_SIZE, self.consts.ENEMY_MISSILE_SIZE)
            new_truck = jnp.where(self.consts.ENABLE_ENEMY_MISSILE_TRUCK_COLLISION, trk2, new_truck)
            new_enemy_mis = mis2

            # Death timers
            new_chop = self.update_entity_death(new_chop, self.consts.FRAMES_DEATH_ANIMATION_ENEMY, False)
            new_jet = self.update_entity_death(new_jet, self.consts.FRAMES_DEATH_ANIMATION_ENEMY, False)
            new_truck = self.update_entity_death(new_truck, self.consts.FRAMES_DEATH_ANIMATION_TRUCK, True)

            # Score / lives
            score = state.score + add_chop + add_jet
            score = jnp.where(pc_jet, score + self.consts.SCORE_PER_JET_KILL, score)
            score = jnp.where(pc_chp, score + self.consts.SCORE_PER_CHOPPER_KILL, score)
            score = jnp.minimum(score, jnp.asarray(999999, dtype=state.score.dtype))

            lives_to_add, save_lives = self.lives_step(player_coll, state.score, state.save_lives)
            lives = state.lives + lives_to_add

            # Align critical dtypes (in addition to the global cast below)
            new_player_mis = new_player_mis.astype(state.player_missile_positions.dtype)
            new_loc_off = new_loc_off.astype(state.local_player_offset.dtype)

            out = state._replace(
                player_x=new_px,
                player_y=new_py,
                player_velocity_x=new_vx,
                local_player_offset=new_loc_off,
                player_facing_direction=new_face,
                score=score,
                lives=lives,
                save_lives=save_lives,
                truck_positions=new_truck,
                jet_positions=new_jet,
                chopper_positions=new_chop,
                enemy_missile_positions=new_enemy_mis,
                player_missile_positions=new_player_mis,
                player_missile_cooldown=new_cd,
                player_collision=player_coll,
                step_counter=state.step_counter + 1,
                rng_key=new_rng
            )
            return _match_state_dtypes(out, state)

        def do_death(_):
            in_pause = (state.pause_timer <= self.consts.DEATH_PAUSE_FRAMES) & (state.pause_timer > 0)
            pt_dtype = state.pause_timer.dtype
            new_pause = jnp.where(in_pause, state.pause_timer - jnp.asarray(1, dtype=pt_dtype), state.pause_timer)
            new_pause = jnp.where(state.lives == 0, jnp.maximum(new_pause, jnp.asarray(1, dtype=pt_dtype)), new_pause)

            # Use normal-state to progress timers/animations
            normal_state = do_normal(None)

            all_dead = jnp.all(state.jet_positions == 0) & jnp.all(state.chopper_positions == 0)

            def trucks_normal():
                temp_truck = self.step_truck_movement(normal_state.truck_positions, normal_state.player_x)
                updated_truck_death_timers = temp_truck[:, 3]
                new_trucks = state.truck_positions.at[:, 3].set(updated_truck_death_timers)
                add_to_score = jnp.asarray(0, dtype=state.score.dtype)
                return new_trucks, add_to_score

            def trucks_points():
                start_idx = 6
                elapsed = self.consts.DEATH_PAUSE_FRAMES - state.pause_timer
                frames_per_kill = self.consts.DEATH_PAUSE_FRAMES // self.consts.MAX_TRUCKS

                num_remove_now = jnp.clip(elapsed // frames_per_kill, 0, self.consts.MAX_TRUCKS)
                num_remove_prev = jnp.clip((elapsed - 1) // frames_per_kill, 0, self.consts.MAX_TRUCKS)

                ids = jnp.arange(self.consts.MAX_TRUCKS)
                rel_idx = jnp.mod(ids - start_idx, self.consts.MAX_TRUCKS)

                kill_mask_now = rel_idx < num_remove_now
                kill_mask_prev = rel_idx < num_remove_prev
                newly_killed = jnp.logical_and(kill_mask_now, jnp.logical_not(kill_mask_prev))

                directions = state.truck_positions[:, 2]
                alive_before = directions != 0
                valid_kill = jnp.logical_and(newly_killed, alive_before)
                num_valid_kill = jnp.sum(valid_kill)

                add_to_score = (num_valid_kill * self.consts.SCORE_PER_TRUCK_ALIVE).astype(state.score.dtype)

                zero_truck = jnp.zeros_like(state.truck_positions)
                new_trucks = jnp.where(kill_mask_now[:, None], zero_truck, state.truck_positions)
                return new_trucks, add_to_score

            new_trucks, add_score = jax.lax.cond(all_dead, lambda _: trucks_points(), lambda _: trucks_normal(), operand=None)
            new_score = state.score + add_score

            # Advance death animation timers
            temp_jet, temp_chop, _ = self.step_enemy_movement(
                normal_state.truck_positions, normal_state.jet_positions, normal_state.chopper_positions,
                normal_state.rng_key, normal_state.player_x, normal_state.local_player_offset,
                normal_state.difficulty, normal_state.enemy_speed,
            )
            new_jet = state.jet_positions.at[:, 3].set(temp_jet[:, 3])
            new_chop = state.chopper_positions.at[:, 3].set(temp_chop[:, 3])

            # Player missile keeps traveling during the pause
            new_player_mis, new_cd = self.player_missile_step(normal_state, normal_state.player_x, normal_state.player_y, action)
            new_player_mis = new_player_mis.astype(state.player_missile_positions.dtype)

            # Immediately delete all enemy missiles
            em_dtype = state.enemy_missile_positions.dtype
            fill_vec = jnp.asarray([0, 0, 0, 187], dtype=em_dtype)
            new_enemy_mis = jnp.full((self.consts.MAX_ENEMY_MISSILES, 4), fill_vec, dtype=em_dtype)

            out = state._replace(
                score=new_score,
                truck_positions=new_trucks,
                jet_positions=new_jet,
                chopper_positions=new_chop,
                enemy_missile_positions=new_enemy_mis,
                player_missile_positions=new_player_mis,
                player_missile_cooldown=new_cd,
                pause_timer=new_pause,
            )
            return _match_state_dtypes(out, state)

        def do_no_move(_):
            no_input = jnp.isin(action, Action.NOOP)

            on_screen = (self.consts.WIDTH // 2) - 8 + state.local_player_offset + (state.player_velocity_x * self.consts.DISTANCE_WHEN_FLYING)
            chopper_at_point = on_screen <= self.consts.ALLOW_MOVE_OFFSET

            lp_dtype = state.local_player_offset.dtype
            offset_speed = jnp.asarray(self.consts.LOCAL_PLAYER_OFFSET_SPEED, dtype=lp_dtype)

            new_lpo = jnp.where(chopper_at_point, state.local_player_offset, state.local_player_offset - offset_speed)

            pt_dtype = state.pause_timer.dtype
            keep_no_move = jnp.asarray(self.consts.DEATH_PAUSE_FRAMES + 1, dtype=pt_dtype)
            new_pause = jnp.where(jnp.logical_and(chopper_at_point, jnp.logical_not(no_input)), keep_no_move, state.pause_timer)

            out = state._replace(local_player_offset=new_lpo, pause_timer=new_pause)
            return _match_state_dtypes(out, state)

        def do_respawn(_):
            # Soft = state during the death pause; we do NOT need a hard reset here
            soft = do_death(None)

            # Only when ALL enemies are dead we spawn new fleets; otherwise we keep them.
            cleared = jnp.logical_and(jnp.all(soft.jet_positions == 0),
                                      jnp.all(soft.chopper_positions == 0))

            # Helper: find the center X of the fleet that lies to the LEFT of the player
            def nearest_left_fleet_center_x(trucks, player_x):
                # Middle trucks per fleet: indices 1, 4, 7, 10 → fetch their X coordinates
                mids = jnp.array([
                    trucks[1, 0],
                    trucks[4, 0],
                    trucks[7, 0],
                    trucks[10, 0],
                ], dtype=trucks.dtype)

                # Candidates wrapped by ±k*1248 so we can find the nearest left copy
                ks = jnp.arange(-3, 4, dtype=jnp.int32)  # [-3 .. +3] is practically sufficient
                candidates = mids[None, :] + ks[:, None].astype(trucks.dtype) * jnp.asarray(1248.0, dtype=trucks.dtype)
                cand_flat = candidates.reshape(-1)

                diffs = player_x - cand_flat  # positive values = candidate lies to the left
                mask_left = diffs > 0

                # Primary: nearest left candidate within 624px
                within = diffs <= jnp.asarray(624.0, dtype=diffs.dtype)
                valid = mask_left & within
                big = jnp.asarray(1e9, dtype=diffs.dtype)

                diffs_valid = jnp.where(valid, diffs, big)
                idx_best = jnp.argmin(diffs_valid)

                # Fallback: if none <= 624px, take the nearest left candidate regardless of distance
                none_valid = jnp.all(~valid)
                diffs_left_only = jnp.where(mask_left, diffs, big)
                idx_fallback = jnp.argmin(diffs_left_only)

                idx_final = jnp.where(none_valid, idx_fallback, idx_best)
                return cand_flat[idx_final]

            # Common player reset fields (for the “keep enemies” path, the new spawn X goes here)
            def _player_reset_fields(s, spawn_x):
                return dict(
                    player_x=spawn_x,
                    player_y=jnp.asarray(self.consts.PLAYER_START_Y, dtype=s.player_y.dtype),
                    player_velocity_x=jnp.asarray(0, dtype=s.player_velocity_x.dtype),
                    local_player_offset=jnp.asarray(50, dtype=s.local_player_offset.dtype),
                    player_facing_direction=jnp.asarray(1, dtype=s.player_facing_direction.dtype),
                    player_missile_positions=jnp.zeros_like(s.player_missile_positions),
                    player_missile_cooldown=jnp.asarray(0, dtype=s.player_missile_cooldown.dtype),
                    player_collision=jnp.asarray(False, dtype=s.player_collision.dtype),
                    pause_timer=jnp.asarray(self.consts.DEATH_PAUSE_FRAMES + 2, dtype=s.pause_timer.dtype),
                    # Remove all enemy missiles
                    enemy_missile_positions=jnp.full(
                        s.enemy_missile_positions.shape,
                        jnp.asarray([0.0, 0.0, 0.0, 187.0], dtype=s.enemy_missile_positions.dtype)
                    ),
                )

            def respawn_keep_enemies():
                # Spawn X: center of the fleet on the left minus 30
                center_left = nearest_left_fleet_center_x(soft.truck_positions, soft.player_x)
                spawn_x = (center_left - jnp.asarray(156.0, dtype=soft.player_x.dtype)).astype(soft.player_x.dtype)

                out = soft._replace(**_player_reset_fields(soft, spawn_x))
                return _match_state_dtypes(out, state)

            def respawn_new_fleets():
                # New fleets deterministically from state.rng_key
                rng_next, rng_init = jax.random.split(soft.rng_key)
                new_trucks = self.initialize_truck_positions().astype(soft.truck_positions.dtype)

                # Spawn X for a new “level”: align to the left of the player at the first (new) fleet
                center_left = nearest_left_fleet_center_x(new_trucks, soft.player_x)
                spawn_x = (center_left + jnp.asarray(156.0, dtype=soft.player_x.dtype)).astype(soft.player_x.dtype)

                # spawn enemies already wrapped near spawn_x
                new_jets, new_choppers = self.initialize_enemy_positions(rng_init, spawn_x)

                out = soft._replace(
                    **_player_reset_fields(soft, spawn_x),
                    jet_positions=new_jets.astype(soft.jet_positions.dtype),
                    chopper_positions=new_choppers.astype(soft.chopper_positions.dtype),
                    truck_positions=new_trucks,
                    enemy_speed=soft.enemy_speed + jnp.asarray(0.5, dtype=soft.enemy_speed.dtype),
                    rng_key=rng_next,
                )
                return _match_state_dtypes(out, state)

            return jax.lax.cond(cleared,
                                lambda _: respawn_new_fleets(),
                                lambda _: respawn_keep_enemies(),
                                operand=None)

        # Switch over phases
        step_state = jax.lax.switch(phase, (do_normal, do_death, do_no_move, do_respawn), operand=None)

        # Initialize/hold death pause
        all_dead  = jnp.all(step_state.jet_positions == 0) & jnp.all(step_state.chopper_positions == 0)
        just_died = step_state.player_collision & (step_state.pause_timer > self.consts.DEATH_PAUSE_FRAMES)
        init_pause = all_dead & (step_state.pause_timer > self.consts.DEATH_PAUSE_FRAMES)
        step_state = step_state._replace(
            pause_timer=jnp.where(just_died | init_pause,
                                  jnp.asarray(self.consts.DEATH_PAUSE_FRAMES, dtype=step_state.pause_timer.dtype),
                                  step_state.pause_timer)
        )

        # Obs/Reward/Done/Info + stack
        observation = self._get_observation(step_state)

        done        = self._get_done(step_state)
        env_reward  = self._get_reward(prev, step_state)
        info        = self._get_info(step_state)

        # Return matching your caller (obs, state, reward, done, info)
        return observation, step_state, env_reward, done, info


class ChopperCommandRenderer(JAXGameRenderer):
    def __init__(self, consts: ChopperCommandConstants = None):
        super().__init__()
        self.consts = consts or ChopperCommandConstants()
        # Use a consistent sprite path based on the old load_sprites function
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.sprite_path = os.path.join(MODULE_DIR, "sprites/choppercommand")
        
        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=1,
            downscale=(84,84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # 2. Get the declarative asset manifest
        asset_config = self._get_asset_config() 
        
        # 3. Make one call to load and process all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, self.sprite_path)
        
        # --- 4. Replicate Animation Stack Logic (from old load_sprites) ---
        # Player Chopper (2 frames -> 6 frames)
        pc_stack = self.SHAPE_MASKS['player_chopper']
        self.SHAPE_MASKS['player_chopper'] = jnp.concatenate([
            jnp.repeat(pc_stack[0][None], 3, axis=0),
            jnp.repeat(pc_stack[1][None], 3, axis=0)
        ])
        
        # Friendly Truck (2 frames -> 8 frames)
        ft_stack = self.SHAPE_MASKS['friendly_truck']
        self.SHAPE_MASKS['friendly_truck'] = jnp.concatenate([
            jnp.repeat(ft_stack[0][None], 4, axis=0),
            jnp.repeat(ft_stack[1][None], 4, axis=0)
        ])
        
        # Enemy Heli (2 frames -> 8 frames)
        eh_stack = self.SHAPE_MASKS['enemy_heli']
        self.SHAPE_MASKS['enemy_heli'] = jnp.concatenate([
            jnp.repeat(eh_stack[0][None], 4, axis=0),
            jnp.repeat(eh_stack[1][None], 4, axis=0)
        ])
        
        # --- 6. Store final sprite animation lengths ---
        self.anim_len = {
            'background_scroll': self.SHAPE_MASKS['background_scroll'].shape[0], # 160
            'friendly_truck': self.SHAPE_MASKS['friendly_truck'].shape[0], # 8
            'enemy_heli': self.SHAPE_MASKS['enemy_heli'].shape[0],       # 8
            'player_chopper': self.SHAPE_MASKS['player_chopper'].shape[0], # 6
            'player_missile': self.SHAPE_MASKS['player_missile'].shape[0], # 16
            'minimap_mountains': self.SHAPE_MASKS['minimap_mountains'].shape[0], # 8
        }

    def _get_asset_config(self) -> list[dict]:
        """
        Returns the declarative asset manifest based on the old load_sprites() function.
        """
        # Load all 160 background files (1.npy to 160.npy)
        bg_scroll_files = [f'bg/{i}.npy' for i in range(1, 161)]
        
        # Player missile animation (missile_0.npy to missile_15.npy)
        pl_missile_files = [f'player_missiles/missile_{i}.npy' for i in range(16)]
        
        # Minimap mountains (1.npy to 8.npy)
        minimap_mountain_files = [f'minimap/mountains/{i}.npy' for i in range(1, 9)]
        
        return [
            # Procedural black background (the sky)
            {'name': 'background', 'type': 'background', 'data': jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 4), dtype=jnp.uint8).at[:,:,3].set(255)},
            # Define the group for all 160 background slices
            {'name': 'background_scroll', 'type': 'group', 'files': bg_scroll_files},
            
            # --- Main Game Sprites ---
            {'name': 'player_chopper', 'type': 'group', 'files': ['player_chopper/1.npy', 'player_chopper/2.npy']},
            {'name': 'friendly_truck', 'type': 'group', 'files': ['friendly_truck/1.npy', 'friendly_truck/2.npy']},
            {'name': 'enemy_jet', 'type': 'single', 'file': 'enemy_jet/normal.npy'},
            {'name': 'enemy_heli', 'type': 'group', 'files': ['enemy_chopper/1.npy', 'enemy_chopper/2.npy']},
            {'name': 'enemy_missile', 'type': 'single', 'file': 'bomb/1.npy'},
            {'name': 'player_missile', 'type': 'group', 'files': pl_missile_files},
            
            # Death animations
            {'name': 'player_death', 'type': 'group', 'files': ['player_chopper/death_1.npy', 'player_chopper/death_2.npy', 'player_chopper/death_3.npy']},
            {'name': 'enemy_death', 'type': 'group', 'files': ['enemy_death/death_1.npy', 'enemy_death/death_2.npy', 'enemy_death/death_3.npy']},
            # --- UI Sprites ---
            {'name': 'digits', 'type': 'digits', 'pattern': 'score/{}.npy'},
            {'name': 'life_indicator', 'type': 'single', 'file': 'score/chopper.npy'},
            # --- Minimap Sprites ---
            {'name': 'minimap_bg', 'type': 'single', 'file': 'minimap/background.npy'},
            {'name': 'minimap_mountains', 'type': 'group', 'files': minimap_mountain_files},
            {'name': 'minimap_truck', 'type': 'single', 'file': 'minimap/truck.npy'},
            {'name': 'minimap_enemy', 'type': 'single', 'file': 'minimap/enemy.npy'},
            {'name': 'minimap_player', 'type': 'single', 'file': 'minimap/player.npy'},
            {'name': 'minimap_logo', 'type': 'single', 'file': 'minimap/activision_logo.npy'},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: ChopperCommandState) -> chex.Array:
        # --- 1. Initialization ---
        # Start with the static black sky
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Calculate shared screen positions
        chopper_position = (self.consts.WIDTH // 2) + state.local_player_offset + (state.player_velocity_x * self.consts.DISTANCE_WHEN_FLYING) - (self.consts.PLAYER_SIZE[0] // 2)
        static_center_x_jet = (self.consts.WIDTH // 2) + state.local_player_offset - (self.consts.JET_SIZE[0] // 2)
        static_center_x_chopper = (self.consts.WIDTH // 2) + state.local_player_offset - (self.consts.CHOPPER_SIZE[0] // 2)
        static_center_x_truck = (self.consts.WIDTH // 2) + state.local_player_offset - (self.consts.TRUCK_SIZE[0] // 2)

        # --- 2. Render Background Scroll (Reverted to original logic) ---
        
        # Calculate the index of the pre-shifted slice to draw
        frame_idx = jnp.asarray(state.local_player_offset + (-state.player_x % self.consts.WIDTH), dtype=jnp.int32)
        frame_idx = jnp.clip(frame_idx, 0, self.anim_len['background_scroll'] - 1)
        
        # Get the pre-shifted background mask from the stack
        bg_mask = self.SHAPE_MASKS['background_scroll'][frame_idx]
        
        # Stamp the full-screen ground mask onto the black sky
        raster = self.jr.render_at(
            raster, 0, 0, 
            bg_mask, 
            flip_offset=self.FLIP_OFFSETS['background_scroll']
        )
        # --- End of Reverted Logic ---

        # --- 3. Render Trucks ---
        truck_anim_idx = state.step_counter % self.anim_len['friendly_truck'] # % 8
        frame_friendly_truck = self.SHAPE_MASKS['friendly_truck'][truck_anim_idx]
        truck_offset = self.FLIP_OFFSETS['friendly_truck']
        
        def render_truck(i, raster_base):
            death_timer = state.truck_positions[i][3]
            direction = state.truck_positions[i][2]
            
            is_alive = (direction != 0) & (death_timer > self.consts.FRAMES_DEATH_ANIMATION_TRUCK)
            is_dying = (direction != 0) & (death_timer <= self.consts.FRAMES_DEATH_ANIMATION_TRUCK) & (death_timer > 0)
            in_flicker_on = (death_timer % self.consts.TRUCK_FLICKER_RATE) < (self.consts.TRUCK_FLICKER_RATE // 2)
            should_render = is_alive | (is_dying & in_flicker_on)
            truck_screen_x = state.truck_positions[i][0] - state.player_x + static_center_x_truck
            truck_screen_y = state.truck_positions[i][1]
            flip_h = (state.truck_positions[i][2] == -1)
            
            return jax.lax.cond(
                should_render,
                lambda r: self.jr.render_at_clipped(
                    r, truck_screen_x, truck_screen_y,
                    frame_friendly_truck, flip_horizontal=flip_h, flip_offset=truck_offset
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_TRUCKS, render_truck, raster)

        # --- 4. Render Enemy Jets ---
        frame_enemy_jet = self.SHAPE_MASKS['enemy_jet'] # Single frame 
        jet_offset = self.FLIP_OFFSETS['enemy_jet']
        death_masks = self.SHAPE_MASKS['enemy_death']
        death_offset = self.FLIP_OFFSETS['enemy_death']
        
        def render_enemy_jet(i, raster_base):
            death_timer = state.jet_positions[i][3]
            should_render = (state.jet_positions[i][2] != 0) & (death_timer > 0)
            jet_screen_x = state.jet_positions[i][0] - state.player_x + static_center_x_jet
            jet_screen_y = state.jet_positions[i][1]
            flip_h = (state.jet_positions[i][2] == -1)
            is_dying = death_timer <= self.consts.FRAMES_DEATH_ANIMATION_ENEMY
            phase0 = death_timer > (2 * self.consts.FRAMES_DEATH_ANIMATION_ENEMY) // 3
            phase1 = (death_timer <= (2 * self.consts.FRAMES_DEATH_ANIMATION_ENEMY) // 3) & \
                     (death_timer > self.consts.FRAMES_DEATH_ANIMATION_ENEMY // 3)
            
            death_sprite_mask = jnp.where(
                phase0, death_masks[0],
                jnp.where(phase1, death_masks[1], death_masks[2])
            )
            
            def render_living(r):
                return self.jr.render_at_clipped(
                    r, jet_screen_x, jet_screen_y, frame_enemy_jet,
                    flip_horizontal=flip_h, flip_offset=jet_offset
                )
            
            def render_dying(r):
                return self.jr.render_at_clipped(
                    r, jet_screen_x, jet_screen_y - 2, death_sprite_mask,
                    flip_horizontal=flip_h, flip_offset=death_offset
                )
            
            return jax.lax.cond(
                should_render,
                lambda r: jax.lax.cond(is_dying, render_dying, render_living, r),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_JETS, render_enemy_jet, raster)

        # --- 5. Render Enemy Choppers ---
        chopper_anim_idx = state.step_counter % self.anim_len['enemy_heli'] # % 8
        frame_enemy_chopper = self.SHAPE_MASKS['enemy_heli'][chopper_anim_idx]
        chopper_offset = self.FLIP_OFFSETS['enemy_heli']
        death_masks = self.SHAPE_MASKS['enemy_death']
        death_offset = self.FLIP_OFFSETS['enemy_death']
        
        def render_enemy_chopper(i, raster_base):
            death_timer = state.chopper_positions[i][3]
            should_render = (state.chopper_positions[i][2] != 0) & (death_timer > 0)
            chopper_screen_x = state.chopper_positions[i][0] - state.player_x + static_center_x_chopper
            chopper_screen_y = state.chopper_positions[i][1]
            flip_h = (state.chopper_positions[i][2] == -1)
            is_dying = death_timer <= self.consts.FRAMES_DEATH_ANIMATION_ENEMY
            phase0 = death_timer > (2 * self.consts.FRAMES_DEATH_ANIMATION_ENEMY) // 3
            phase1 = (death_timer <= (2 * self.consts.FRAMES_DEATH_ANIMATION_ENEMY) // 3) & \
                     (death_timer > self.consts.FRAMES_DEATH_ANIMATION_ENEMY // 3)
            
            death_sprite_mask = jnp.where(
                phase0, death_masks[0],
                jnp.where(phase1, death_masks[1], death_masks[2])
            )
            final_mask = jnp.where(is_dying, death_sprite_mask, frame_enemy_chopper)
            final_offset = jnp.where(is_dying, death_offset, chopper_offset)
            
            return jax.lax.cond(
                should_render,
                lambda r: self.jr.render_at_clipped(
                    r, chopper_screen_x, chopper_screen_y, final_mask,
                    flip_horizontal=flip_h, flip_offset=final_offset
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_CHOPPERS, render_enemy_chopper, raster)

        # --- 6. Render Enemy Missiles ---
        frame_enemy_missile = self.SHAPE_MASKS['enemy_missile'] # Single mask
        missile_offset = self.FLIP_OFFSETS['enemy_missile']
        
        def render_enemy_missile(i, raster_base):
            should_render = state.enemy_missile_positions[i][1] > 2
            x_pos = state.enemy_missile_positions[i][0] - state.player_x + static_center_x_chopper
            y_pos = state.enemy_missile_positions[i][1]
            
            return jax.lax.cond(
                should_render,
                lambda r: self.jr.render_at_clipped(
                    r, x_pos, y_pos, frame_enemy_missile, flip_offset=missile_offset
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_MISSILES, render_enemy_missile, raster)

        # --- 7. Render Score ---
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        
        # Find first non-zero digit
        is_zero = jnp.all(score_digits == 0)
        first_nonzero = jnp.argmax(score_digits != 0)
        start_idx = jax.lax.select(is_zero, 5, first_nonzero) # If 0, show last digit
        num_to_render = 6 - start_idx
        x_start = 16 + start_idx * 8 # 8 is spacing
        raster = self.jr.render_label_selective(
            raster, x_start, 2, score_digits, self.SHAPE_MASKS['digits'],
            start_idx, num_to_render, spacing=8, max_digits_to_render=6
        )
        
        # --- 8. Render Lives ---
        raster = self.jr.render_indicator(
            raster, 16, 10, state.lives - 1, 
            self.SHAPE_MASKS['life_indicator'], 
            spacing=9, max_value=5 # Assuming max 5 lives displayed
        )

        # --- 9. Render Player ---
        player_anim_idx = state.step_counter % self.anim_len['player_chopper'] # % 6
        frame_pl_heli = self.SHAPE_MASKS['player_chopper'][player_anim_idx]
        player_offset = self.FLIP_OFFSETS['player_chopper']
        death_timer = state.pause_timer
        should_render = (death_timer != 0) & (death_timer != 1)
        phase0_cutoff = jnp.asarray(self.consts.PLAYER_FADE_OUT_START_THRESHOLD_0 * self.consts.DEATH_PAUSE_FRAMES, dtype=jnp.int32)
        phase1_cutoff = jnp.asarray(self.consts.PLAYER_FADE_OUT_START_THRESHOLD_1 * self.consts.DEATH_PAUSE_FRAMES, dtype=jnp.int32)
        phase0 = death_timer > phase0_cutoff
        phase1 = (death_timer <= phase0_cutoff) & (death_timer > phase1_cutoff)
        player_death_masks = self.SHAPE_MASKS['player_death']
        death_sprite = jnp.where(
            phase0, player_death_masks[0],
            jnp.where(phase1, player_death_masks[1], player_death_masks[2])
        )
        death_sprite_offset = self.FLIP_OFFSETS['player_death']
        all_enemies_dead = jnp.all(state.jet_positions == 0) & jnp.all(state.chopper_positions == 0)
        is_alive = (death_timer > self.consts.DEATH_PAUSE_FRAMES) | all_enemies_dead
        
        final_player_mask = jnp.where(is_alive, frame_pl_heli, death_sprite)
        final_player_offset = jnp.where(is_alive, player_offset, death_sprite_offset)
        
        raster = jax.lax.cond(
            should_render,
            lambda r: self.jr.render_at(
                r, chopper_position, state.player_y, final_player_mask,
                flip_horizontal=(state.player_facing_direction == -1),
                flip_offset=final_player_offset
            ),
            lambda r: r,
            raster,
        )

        # --- 10. Render Player Missiles ---
        missile_stack = self.SHAPE_MASKS['player_missile']
        missile_stack_offset = self.FLIP_OFFSETS['player_missile']
        
        def render_single_missile(i, r):
            missile = state.player_missile_positions[i]
            missile_active = missile[2] != 0
            
            missile_screen_x = missile[0] - state.player_x + chopper_position
            missile_screen_y = missile[1]
            delta_spawn = jnp.abs(missile[0] - missile[3])
            frame_idx = jnp.floor_divide(delta_spawn, self.consts.MISSILE_ANIMATION_SPEED)
            frame_idx = jnp.clip(frame_idx, 0, self.anim_len['player_missile'] - 1).astype(jnp.int32) # Clip to 15
            frame_pl_missile = missile_stack[frame_idx]
            
            return jax.lax.cond(
                missile_active,
                lambda r_in: self.jr.render_at_clipped(
                    r_in, missile_screen_x, missile_screen_y,
                    frame_pl_missile,
                    flip_horizontal=(missile[2] == -1),
                    flip_offset=missile_stack_offset
                ),
                lambda r_in: r_in,
                r,
            )

        raster = jax.lax.fori_loop(0, state.player_missile_positions.shape[0], render_single_missile, raster)

        # --- 11. Render Minimap ---
        raster = self.render_minimap(chopper_position, raster, state)

        # --- 12. Final Palette Lookup ---
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def render_minimap(self, chopper_position, raster, state: ChopperCommandState):
        # Render minimap background
        raster = self.jr.render_at(
            raster, self.consts.MINIMAP_POSITION_X, self.consts.MINIMAP_POSITION_Y,
            self.SHAPE_MASKS['minimap_bg'], flip_offset=self.FLIP_OFFSETS['minimap_bg']
        )

        # Render minimap mountains
        frame_idx = jnp.asarray(((-state.player_x // (self.consts.DOWNSCALING_FACTOR_WIDTH * 7)) % 8), dtype=jnp.int32)
        frame_idx = jnp.clip(frame_idx, 0, self.anim_len['minimap_mountains'] - 1) # Clip to 7
        frame_minimap_mountains = self.SHAPE_MASKS['minimap_mountains'][frame_idx]
        raster = self.jr.render_at(
            raster, self.consts.MINIMAP_POSITION_X, self.consts.MINIMAP_POSITION_Y + 3,
            frame_minimap_mountains, flip_offset=self.FLIP_OFFSETS['minimap_mountains']
        )

        # Render trucks on minimap
        minimap_truck_mask = self.SHAPE_MASKS['minimap_truck']
        minimap_truck_offset = self.FLIP_OFFSETS['minimap_truck']
        
        def render_trucks_minimap(i, raster_base):
            timing_clock = state.step_counter % self.consts.MINIMAP_RENDER_TRUCK_REFRESH_RATE
            update_trigger = timing_clock == 0
            truck_world_x = state.truck_positions[i][0] # Use index i
            truck_world_x = jnp.where(
                update_trigger, truck_world_x,
                truck_world_x + (0.5 * timing_clock)
            )
            
            weird_offset = 16
            parent_x = weird_offset + ((truck_world_x - state.player_x + chopper_position) // self.consts.DOWNSCALING_FACTOR_WIDTH // 6)
            minimap_x = 2 * i + parent_x
            add = 11.5
            is_in_first_fleet = i < 3
            is_in_second_fleet = (i >= 3) & (i <= 5)
            is_in_third_fleet = (i >= 6) & (i <= 8)
            
            add_to_fleet = jnp.where(is_in_first_fleet, -3 * add,
                             jnp.where(is_in_second_fleet, -2 * add,
                             jnp.where(is_in_third_fleet, -1 * add, 0.0)))
            minimap_x = jnp.mod(minimap_x + add_to_fleet, 6 * add)
            minimap_y = (state.truck_positions[i][1] // self.consts.DOWNSCALING_FACTOR_HEIGHT)
            
            should_render = (state.truck_positions[i][3] != 0) & \
                            (minimap_x >= 0) & (minimap_x < self.consts.MINIMAP_WIDTH)
            return jax.lax.cond(
                should_render,
                lambda r: self.jr.render_at(
                    r, self.consts.MINIMAP_POSITION_X + minimap_x,
                    self.consts.MINIMAP_POSITION_Y + 1 + minimap_y,
                    minimap_truck_mask, flip_offset=minimap_truck_offset
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_TRUCKS, render_trucks_minimap, raster)

        # Render jets on minimap
        minimap_enemy_mask = self.SHAPE_MASKS['minimap_enemy']
        minimap_enemy_offset = self.FLIP_OFFSETS['minimap_enemy']
        
        def render_jets_minimap(i, raster_base):
            weird_offset = 16
            jet_world_x = state.jet_positions[i][0]
            minimap_x = weird_offset + ((jet_world_x - state.player_x + chopper_position) // self.consts.DOWNSCALING_FACTOR_WIDTH // 6)
            is_alive = state.jet_positions[i][3] > self.consts.FRAMES_DEATH_ANIMATION_ENEMY
            should_render = is_alive & (minimap_x >= 0) & (minimap_x < self.consts.MINIMAP_WIDTH - 1)
            
            def do_render(r):
                jet_world_y = state.jet_positions[i][1]
                is_in_top_lane = (jet_world_y <= self.consts.ENEMY_LANE_6) & (jet_world_y >= self.consts.ENEMY_LANE_8)
                is_in_middle_lane = (jet_world_y <= self.consts.ENEMY_LANE_3) & (jet_world_y >= self.consts.ENEMY_LANE_5)
                lane_world_y = jnp.where(is_in_top_lane, self.consts.ENEMY_LANE_7,
                                 jnp.where(is_in_middle_lane, self.consts.ENEMY_LANE_4, self.consts.ENEMY_LANE_1))
                minimap_y = (lane_world_y // (self.consts.DOWNSCALING_FACTOR_HEIGHT + 1))
                return self.jr.render_at(
                    r, self.consts.MINIMAP_POSITION_X + minimap_x,
                    self.consts.MINIMAP_POSITION_Y + 3 + minimap_y,
                    minimap_enemy_mask, flip_offset=minimap_enemy_offset
                )
            
            return jax.lax.cond(should_render, do_render, lambda r: r, raster_base)

        raster = jax.lax.fori_loop(0, self.consts.MAX_JETS, render_jets_minimap, raster)

        # Render choppers on minimap
        def render_choppers_minimap(i, raster_base):
            weird_offset = 16
            chopper_world_x = state.chopper_positions[i][0]
            minimap_x = weird_offset + ((chopper_world_x - state.player_x + chopper_position) // self.consts.DOWNSCALING_FACTOR_WIDTH // 6)
            is_alive = state.chopper_positions[i][3] > self.consts.FRAMES_DEATH_ANIMATION_ENEMY
            should_render = is_alive & (minimap_x >= 0) & (minimap_x < self.consts.MINIMAP_WIDTH - 1)
            
            def do_render(r):
                chopper_world_y = state.chopper_positions[i][1]
                is_in_top_lane = (chopper_world_y <= self.consts.ENEMY_LANE_6) & (chopper_world_y >= self.consts.ENEMY_LANE_8)
                is_in_middle_lane = (chopper_world_y <= self.consts.ENEMY_LANE_3) & (chopper_world_y >= self.consts.ENEMY_LANE_5)
                lane_world_y = jnp.where(is_in_top_lane, self.consts.ENEMY_LANE_7,
                                 jnp.where(is_in_middle_lane, self.consts.ENEMY_LANE_4, self.consts.ENEMY_LANE_1))
                minimap_y = (lane_world_y // (self.consts.DOWNSCALING_FACTOR_HEIGHT + 1))
                return self.jr.render_at(
                    r, self.consts.MINIMAP_POSITION_X + minimap_x,
                    self.consts.MINIMAP_POSITION_Y + 3 + minimap_y,
                    minimap_enemy_mask, flip_offset=minimap_enemy_offset
                )
            
            return jax.lax.cond(should_render, do_render, lambda r: r, raster_base)

        raster = jax.lax.fori_loop(0, self.consts.MAX_CHOPPERS, render_choppers_minimap, raster)

        # Render player on minimap
        raster = self.jr.render_at(
            raster,
            self.consts.MINIMAP_POSITION_X + 16 + (chopper_position // (self.consts.DOWNSCALING_FACTOR_WIDTH * 7)),
            self.consts.MINIMAP_POSITION_Y + 6 + (state.player_y // (self.consts.DOWNSCALING_FACTOR_HEIGHT + 7)),
            self.SHAPE_MASKS['minimap_player'], flip_offset=self.FLIP_OFFSETS['minimap_player']
        )

        # Render activision logo
        raster = self.jr.render_at(
            raster,
            self.consts.MINIMAP_POSITION_X + (self.consts.MINIMAP_WIDTH - 32) // 2,
            self.consts.HEIGHT_ONLY_PLAYING_FIELD - 7 - 1,
            self.SHAPE_MASKS['minimap_logo'], flip_offset=self.FLIP_OFFSETS['minimap_logo']
        )

        return raster