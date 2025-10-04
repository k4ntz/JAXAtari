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


class TurmoilConstants(NamedTuple):
    # pre-defined movement lanes
    VERTICAL_LANE = 76 # x value
    HORIZONTAL_LANES = [37, 58, 79, 100, 121, 142, 163] # y values

    # boundaries
    MIN_BOUND = (2, HORIZONTAL_LANES[0]) # (min x, min y)
    MAX_BOUND = (150, HORIZONTAL_LANES[-1])

    # directions
    FACE_LEFT = -1
    FACE_RIGHT = 1   

    # sizes
    HORIZONTAL_LANE_GAP_SIZE = 17
    PLAYER_SIZE = (8, 11) # (width, height)
    BULLET_SIZE = (8, 3)
    ENEMY_SIZE = (
        (8, 5), # lines
        (8, 10), # arrow
        (8, 13), # tank
        (8, 11), # L
        (7, 13), # T
        (8, 7), # rocket
        (8, 13), # triangle_hollow
        (8, 11), # x_shape
        (4, 5), # sonic boom
    )
    ENEMY_SIZE_FOR_COLLISION = (8, 13)  # max values from ENEMY_SIZE
    PRIZE_SIZE = (8, 9)

    # y offsets for finding the middle of the lane for each sprite
    Y_OFFSET_PLAYER = (HORIZONTAL_LANE_GAP_SIZE - PLAYER_SIZE[1]) // 2
    Y_OFFSET_PRIZE = (HORIZONTAL_LANE_GAP_SIZE - PRIZE_SIZE[1]) // 2
    Y_OFFSET_ENEMY = (
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[0][1]) // 2, # lines
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[1][1]) // 2, # arrow
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[2][1]) // 2, # tank
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[3][1]) // 2, # L
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[4][1]) // 2, # T
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[5][1]) // 2, # rocket
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[6][1]) // 2, # triangle_hollow
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[7][1]) // 2, # x_shape
        (HORIZONTAL_LANE_GAP_SIZE - ENEMY_SIZE[8][1]) // 2, # sonic boom
    )

    # player
    PLAYER_START_POS = (VERTICAL_LANE, HORIZONTAL_LANES[6] + Y_OFFSET_PLAYER) # (starting_x_pos, starting_y_pos)
    PLAYER_STEP_COOLDOWN = (0, 5) # (x cooldown, y cooldown)
    PLAYER_SPEED = (1, 21) # (x_step_size, y_step_size)
    PLAYER_X_LANE_MOVEMENT_BUFFER = PLAYER_SIZE[1] // 2 # +/- this amount from VERTICAL_LANE player can move vertically

    # bullet
    BULLET_SPEED = 5

    # enemy types, so it is easier to identify by name
    ENEMY_TYPES = {
        "3lines" : 0,
        "arrow" : 1,
        "tank" : 2,
        "L" : 3,
        "T" : 4,
        "rocket" : 5,
        "triangle_hollow" : 6,
        "x_shape" : 7,
        "boom" : 8
    }

    ENEMY_SPEED = (
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # lines
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # arrow
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # tank
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # L
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # T
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # rocket
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # triangle_hollow
        [1, 1, 1, 1, 2, 2, 2, 2, 3], # x_shape
        [5, 5, 5, 5, 7, 7, 7, 7, 9], # sonic boom
    )

    ENEMY_LEFT_SPAWN_X = 3
    ENEMY_RIGHT_SPAWN_X = 148
    TANK_PUSH_BACK = 5 # tank displacement if shot from front

    # probability of spawning when there is slot available for each level
    ENEMY_SPAWN_PROBABILITY = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17]
    PRIZE_SPAWN_PROBABILITY = [0.005, 0.007, 0.009, 0.011, 0.015, 0.017, 0.019, 0.021, 0.023]
    
    # prize
    PRIZE_TO_BOOM_TIME = 300
    PRIZE_SCORE_REWARD = 800
    PRIZE_TILL_TRIANGLE_HOLLOW_TIMER = 30

    # game phases and control
    STEP_COUNTER = 1024
    LOADING_GAME_PHASE_TIME = 50
    PLAYER_SHRINK_TIME = 120
    LVL_CHANGE_SCORES = ( # lvl end scores
        5000,  # lvl 1
        10000, # lvl 2
        15000, # lvl 3
        20000, # lvl 4
        30000, # lvl 5
        40000, # lvl 6
        60000, # lvl 7
        80000, # lvl 8
        99999  # lvl 9
    )
    BG_APPER_PROBABILITY = 0.4 # after lvl 4, prob. of seeing lanes


class TurmoilState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_step_cooldown: chex.Array # (2,) x_cooldown, y_cooldown
    player_move_unlock: chex.Array # (2,) player to move when off center, cuase of prize
                                   # can_move, position (1 -> right, -1 -> left)
    player_shrink: chex.Array # if player is in shrink mode

    ships: chex.Array # maximum 6 ships
    score: chex.Array
    bullet: chex.Array # x, y, active, direction

    enemy: chex.Array # (7, 7) 7 lanes; 7 -> type (see constants), x, y, active, speed, direction, change_type_coordinate,
    prize: chex.Array # (6,) lane, x, y, active, boom_timer, direction
    spawn_triangle_hollow: chex.Array # (2,) spawn, counter, lane, direction

    game_phase: chex.Array # game phase 0-2
                           # 0 -> loading screen, 1 -> game, 2 -> player shrink
    game_phase_timer: chex.Array

    level: chex.Array # maximim lvl is 9
    step_counter: chex.Array
    rng_key: chex.PRNGKey

    bg_visible: chex.Array # if background is visible


class PlayerEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    direction: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray


class TurmoilObservation(NamedTuple):
    player: PlayerEntity
    ships: jnp.array
    enemy: jnp.array # (7, 7)
    prize: jnp.array # (6,)
    score: jnp.array
    bullet: EntityPosition
    game_phase: jnp.array
    level: jnp.array

class TurmoilInfo(NamedTuple):
    step_counter: jnp.ndarray  # Current step count
    all_rewards: jnp.ndarray  # All rewards for the current step
    level: jnp.ndarray # current game level



# RENDER CONSTANTS
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/bg/1.npy"))
    player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/1.npy"))
    bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/bullet/1.npy"))
    player_shrink_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/1.npy"))
    player_shrink_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/2.npy"))
    player_shrink_3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/3.npy"))
    player_shrink_4 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/4.npy"))
    player_shrink_5 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/5.npy"))
    player_shrink_6 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/6.npy"))

    # enemies
    lines_enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/3lines/1.npy"))
    arrow_enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/arrow/1.npy"))
    boom_enemy_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/boom/1.npy"))
    boom_enemy_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/boom/2.npy"))
    L_enemy_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/L/1.npy"))
    L_enemy_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/L/2.npy"))
    rocket_enemy_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/rocket/1.npy"))
    rocket_enemy_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/rocket/2.npy"))
    T_enemy_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/T/1.npy"))
    T_enemy_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/T/2.npy"))
    tank_enemy_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/tank/1.npy"))
    tank_enemy_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/tank/2.npy"))
    triangle_hollow_enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/triangle_hollow/1.npy"))
    x_shape_enemy_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/x_shape/1.npy"))
    x_shape_enemy_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/enemy/x_shape/2.npy"))


    player = [player]
    bullet = [bullet]
    bg = [bg]
    
    # Pad player shrink sprites to match each other
    player_shrink_sprites, player_shrink_offsets = jr.pad_to_match([
        player_shrink_1,
        player_shrink_2,
        player_shrink_3,
        player_shrink_4,
        player_shrink_5,
        player_shrink_6
    ])
    player_shrink_offsets = jnp.array(player_shrink_offsets)

    lines_enemy_sprites = [lines_enemy]
    arrow_enemy_sprites = [arrow_enemy]
    boom_enemy_sprites, _ = jr.pad_to_match([boom_enemy_1, boom_enemy_2])
    L_enemy_sprites, _ = jr.pad_to_match([L_enemy_1, L_enemy_2])
    rocket_enemy_sprites, _ = jr.pad_to_match([rocket_enemy_1, rocket_enemy_2])
    T_enemy_sprites, _ = jr.pad_to_match([T_enemy_1, T_enemy_2])
    tank_enemy_sprites, _ = jr.pad_to_match([tank_enemy_1, tank_enemy_2])
    triangle_hollow_enemy_sprites = [triangle_hollow_enemy]
    x_shape_enemy_sprites, _ = jr.pad_to_match([x_shape_enemy_1, x_shape_enemy_2])

    # bg sprites
    SPRITE_BG = jnp.repeat(bg[0][None], 1, axis=0)

    # Player sprites
    PLAYER_SHIP = jnp.repeat(player[0][None], 1, axis=0)

    # bullet sprites
    BULLET = jnp.repeat(bullet[0][None], 1, axis=0)

    # player shrink sprites
    PLAYER_SHRINK = jnp.concatenate(
        [
            jnp.repeat(player_shrink_sprites[0][None], 31, axis=0),
            jnp.repeat(player_shrink_sprites[1][None], 5, axis=0),
            jnp.repeat(player_shrink_sprites[2][None], 7, axis=0),
            jnp.repeat(player_shrink_sprites[3][None], 5, axis=0),
            jnp.repeat(player_shrink_sprites[4][None], 7, axis=0),
            jnp.repeat(player_shrink_sprites[5][None], 23, axis=0),
            jnp.repeat(player_shrink_sprites[5][None], 23, axis=0),
            jnp.repeat(player_shrink_sprites[4][None], 7, axis=0),
            jnp.repeat(player_shrink_sprites[3][None], 5, axis=0),
            jnp.repeat(player_shrink_sprites[2][None], 7, axis=0),
        ]
    )

    DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/turmoil/digits/{}.npy"))

    # enemeis
    LINES_ENEMY = jnp.repeat(lines_enemy_sprites[0][None], 1, axis=0)

    ARROW_ENEMY = jnp.repeat(arrow_enemy_sprites[0][None], 1, axis=0)

    BOOM_ENEMY = jnp.repeat(boom_enemy_sprites[0][None], 1, axis=0)

    PRIZE = jnp.concatenate(
        [
            jnp.repeat(boom_enemy_sprites[0][None], 8, axis=0),
            jnp.repeat(boom_enemy_sprites[1][None], 8, axis=0),
        ]
    )

    L_ENEMY = jnp.concatenate(
        [
            jnp.repeat(L_enemy_sprites[0][None], 8, axis=0),
            jnp.repeat(L_enemy_sprites[1][None], 8, axis=0),
        ]
    )

    ROCKET_ENEMY = jnp.concatenate(
        [
            jnp.repeat(rocket_enemy_sprites[0][None], 8, axis=0),
            jnp.repeat(rocket_enemy_sprites[1][None], 8, axis=0),
        ]
    )

    T_ENEMY = jnp.concatenate(
        [
            jnp.repeat(T_enemy_sprites[0][None], 8, axis=0),
            jnp.repeat(T_enemy_sprites[1][None], 8, axis=0),
        ]
    )

    TANK_ENEMY = jnp.concatenate(
        [
            jnp.repeat(tank_enemy_sprites[0][None], 8, axis=0),
            jnp.repeat(tank_enemy_sprites[1][None], 8, axis=0),
        ]
    )

    TRIANGLE_HOLLOW_ENEMY = jnp.repeat(triangle_hollow_enemy_sprites[0][None], 1, axis=0)

    X_SHAPE_ENEMY = jnp.concatenate(
        [
            jnp.repeat(x_shape_enemy_sprites[0][None], 8, axis=0),
            jnp.repeat(x_shape_enemy_sprites[1][None], 8, axis=0),
        ]
    )


    return (
        SPRITE_BG,
        PLAYER_SHIP,
        BULLET,
        PLAYER_SHRINK,
        DIGITS,
        LINES_ENEMY,
        ARROW_ENEMY,
        BOOM_ENEMY,
        L_ENEMY,
        ROCKET_ENEMY,
        T_ENEMY,
        TANK_ENEMY,
        TRIANGLE_HOLLOW_ENEMY,
        X_SHAPE_ENEMY,
        PRIZE,
        player_shrink_offsets
    )

# Load sprites once at module level
(
    SPRITE_BG,
    PLAYER_SHIP,
    BULLET,
    PLAYER_SHRINK,
    DIGITS,
    LINES_ENEMY,
    ARROW_ENEMY,
    BOOM_ENEMY,
    L_ENEMY,
    ROCKET_ENEMY,
    T_ENEMY,
    TANK_ENEMY,
    TRIANGLE_HOLLOW_ENEMY,
    X_SHAPE_ENEMY,
    PRIZE,
    PLAYER_SHRINK_OFFSETS
) = load_sprites()


class JaxTurmoil(JaxEnvironment[TurmoilState, TurmoilObservation, TurmoilInfo, TurmoilConstants]):
    def __init__(self, consts: TurmoilConstants = None, reward_funcs: list[callable] = None):
        consts = consts or TurmoilConstants()
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
        self.obs_size = 6 + 1 + 7 * 7 + 6 * 1 + 1 + 5 + 1 + 1
        self.renderer = TurmoilRenderer(self.consts)

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
            jnp.array([entity.direction], dtype=jnp.int32),
            jnp.array([entity.width], dtype=jnp.int32),
            jnp.array([entity.height], dtype=jnp.int32),
            jnp.array([entity.active], dtype=jnp.int32)
        ])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TurmoilObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_entity(obs.player),
            obs.ships.flatten().astype(jnp.int32),
            obs.enemy.flatten().astype(jnp.int32),
            obs.prize.flatten().astype(jnp.int32),
            obs.score.flatten().astype(jnp.int32),
            self.flatten_entity_position(obs.bullet),
            obs.game_phase.flatten().astype(jnp.int32),
            obs.level.flatten().astype(jnp.int32),
        ])
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TurmoilState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    
    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Seaquest.
        The observation contains:
        - player: PlayerEntity (x, y, direction, width, height, active)
        - enemy: array of shape (7, 7) with type, x, y, active, speed, direction, change_type_coordinate
        - prize: array of shape (6,) with lane, x, y, active, boom_timer, direction
        - ships: int (0-6)
        - score: int (0-999999)
        - bullet: EntityPosition (x, y, width, height, active)
        - game_phase: int (0-2)
        - level: int (0-10) # max lvl is 9, lvl 10 for ending condition
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "ships": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
            "enemy": spaces.Box(low=-1, high=210, shape=(7, 7), dtype=jnp.int32),
            "prize": spaces.Box(low=-1, high=300, shape=(6,), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "bullet": spaces.Dict({
                "x": spaces.Box(low=-100, high=200, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "game_phase": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=0, high=10, shape=(), dtype=jnp.int32),
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
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TurmoilState) -> TurmoilObservation:
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            direction=state.player_direction,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
            active=jnp.array(1), # always active
        )

        # bullet
        bullet = EntityPosition(
            x=state.bullet[0],
            y=state.bullet[1],
            width=jnp.array(self.consts.BULLET_SIZE[0]),
            height=jnp.array(self.consts.BULLET_SIZE[1]),
            active=jnp.array(state.bullet[2] != 0),
        )

        # convert enemies
        def convert_enemy(e):
            return jnp.array([
                e[1],  # x
                e[2],  # y
                self.consts.ENEMY_SIZE_FOR_COLLISION[0],  # width
                self.consts.ENEMY_SIZE_FOR_COLLISION[1],  # height
                e[3] != 0,  # active
            ])

        enemy= jax.vmap(convert_enemy)(state.enemy)

        # prize
        prize = jnp.array([
            state.prize[1], # x
            state.prize[2], # y
            self.consts.PRIZE_SIZE[0], # width
            self.consts.PRIZE_SIZE[1], # height
            state.prize[3] != 0 # active
        ])

        return TurmoilObservation(
            player=player,
            ships=state.ships,
            enemy=enemy,
            prize=prize,
            score=state.score,
            bullet=bullet,
            game_phase=state.game_phase,
            level=state.level,
        )
    

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TurmoilState, all_rewards: jnp.ndarray = None) -> TurmoilInfo:
        return TurmoilInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
            level=state.level,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TurmoilState, state: TurmoilState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: TurmoilState, state: TurmoilState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TurmoilState) -> bool:
        return jnp.logical_or(state.ships <= 0, state.level >= 10)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[TurmoilObservation, TurmoilState]:
        """Initialize game state"""
        reset_state = TurmoilState(
            player_x=jnp.array(self.consts.PLAYER_START_POS[0]),
            player_y=jnp.array(self.consts.PLAYER_START_POS[1]),
            player_direction=jnp.array(0),
            player_step_cooldown=jnp.zeros(2),
            player_move_unlock=jnp.zeros(2),
            player_shrink=jnp.array(0),
            ships=jnp.array(5),
            score=jnp.array(0),

            bullet=jnp.zeros(4),

            enemy=jnp.zeros((7, 7)),
            prize=jnp.zeros(6),
            spawn_triangle_hollow=jnp.zeros(4),

            game_phase=jnp.array(0),
            game_phase_timer=jnp.array(0),
            level=jnp.array(1),
            step_counter=jnp.array(0),
            rng_key=key,
            bg_visible=jnp.array(1),
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state: TurmoilState,
        action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        '''
        implement all the possible movement directions for the player, the mapping is:
        anything with left in it, add -2 to the x position
        anything with right in it, add 2 to the x position
        anything with up in it, add -2 to the y position
        anything with down in it, add 2 to the y position
        '''        
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

        def can_move_horizontal(state: TurmoilState) :
            return jnp.logical_or(
                jnp.logical_and(
                    state.prize[3] == 1, # prize active and player on same lane
                    state.player_y - self.consts.Y_OFFSET_PLAYER == state.prize[2] - self.consts.Y_OFFSET_PRIZE
                ),
                state.player_move_unlock[0] == 1
            )
        
        def can_move_vertical(player_x) :
            return jnp.logical_and(
                player_x >= self.consts.VERTICAL_LANE - self.consts.PLAYER_X_LANE_MOVEMENT_BUFFER,
                player_x <= self.consts.VERTICAL_LANE + self.consts.PLAYER_X_LANE_MOVEMENT_BUFFER,
            )
        
        # cooldown so player does not go too fast
        player_step_cooldown = jnp.where(
            state.player_step_cooldown > 0,
            state.player_step_cooldown - 1,
            0
        )

        # move player
        player_x = jnp.where(
            jnp.logical_and(player_step_cooldown[0] <= 0, can_move_horizontal(state)),
            jnp.where(
                right,
                state.player_x + self.consts.PLAYER_SPEED[0],
                jnp.where(
                    left,
                    state.player_x - self.consts.PLAYER_SPEED[0],
                    state.player_x
                )
            ),
            state.player_x
        )

        player_y = jnp.where(
            jnp.logical_and(player_step_cooldown[1] <= 0, can_move_vertical(player_x)),
            jnp.where(
                down,
                state.player_y + self.consts.PLAYER_SPEED[1],
                jnp.where(
                    up,
                    state.player_y - self.consts.PLAYER_SPEED[1],
                    state.player_y
                )
            ),
            state.player_y
        )

        # if moved vertical set player to VERTICAL_LANE
        moved_vertically = jnp.logical_and(
            jnp.logical_and(player_step_cooldown[1] <= 0, can_move_vertical(player_x)),
            jnp.logical_or(up, down)
        )

        player_x = jnp.where(
            moved_vertically,
            self.consts.VERTICAL_LANE,
            player_x
        )

        player_direction = jnp.where(
            right,
            1,
            jnp.where(
                left,
                -1,
                state.player_direction
            )
        )

        # right/left cooldown
        player_step_cooldown = player_step_cooldown.at[0].set(
            jnp.where(
                jnp.logical_and(
                    jnp.logical_or(left, right),
                    player_step_cooldown[0] <= 0 # if not already on cooldown
                ),
                self.consts.PLAYER_STEP_COOLDOWN[0],
                player_step_cooldown[0],
            )
        )

        # up/down cool down
        player_step_cooldown = player_step_cooldown.at[1].set(
            jnp.where(
                jnp.logical_and(
                    jnp.logical_or(up, down),
                    player_step_cooldown[1] <= 0
                ),
                self.consts.PLAYER_STEP_COOLDOWN[1],
                player_step_cooldown[1],
            )
        )

        # keep player in boundaries
        player_x = jnp.where(
            player_x <= self.consts.MIN_BOUND[0],
            self.consts.MIN_BOUND[0],
            jnp.where(
                player_x >= self.consts.MAX_BOUND[0],
                self.consts.MAX_BOUND[0],
                player_x,
            ),
        )

        player_y = jnp.where(
            player_y <= self.consts.MIN_BOUND[1],
            self.consts.MIN_BOUND[1] + self.consts.Y_OFFSET_PLAYER,
            jnp.where(
                player_y >= self.consts.MAX_BOUND[1],
                self.consts.MAX_BOUND[1] + self.consts.Y_OFFSET_PLAYER,
                player_y
            ),
        )

        # player can move horizontal, if off center
        new_player_move_unlock = jnp.where(
            jnp.logical_and(player_x != self.consts.VERTICAL_LANE, state.player_move_unlock[0] == 0),
            jnp.array([
                1,
                jnp.where(
                    player_x - self.consts.VERTICAL_LANE > 0,
                    1,
                    -1
                )
                
            ]),
            state.player_move_unlock
        )
    
        passed_vertical_lane = jnp.logical_or(
            jnp.logical_and(new_player_move_unlock[1] == 1, player_x <= self.consts.VERTICAL_LANE),
            jnp.logical_and(new_player_move_unlock[1] == -1, player_x >= self.consts.VERTICAL_LANE)
        )

        # player snaps to vertical lane 
        player_x = jnp.where(
            jnp.logical_and(passed_vertical_lane, new_player_move_unlock[0] == 1),
            self.consts.VERTICAL_LANE,
            player_x
        )

        # reset the lock
        new_player_move_unlock = jnp.where(
            jnp.logical_or(new_player_move_unlock[0] == 0, passed_vertical_lane),
            jnp.array([0, 0]),
            new_player_move_unlock
        )

        return player_x, player_y, player_direction, player_step_cooldown, new_player_move_unlock


    @partial(jax.jit, static_argnums=(0,))
    def bullet_step(
        self,
        state: TurmoilState,
        action: chex.Array,
    ) -> chex.Array:
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

        # if player fired and there is no active bullet, create on in player_direction
        new_bullet = jnp.where(
            jnp.logical_and(jnp.logical_and(fire, jnp.logical_not(state.bullet[2])), jnp.logical_not(state.game_phase == 2)),
            jnp.where(
                state.player_direction == -1,
                jnp.array([
                    state.player_x - self.consts.BULLET_SIZE[0], # x, y, active, direction
                    state.player_y + self.consts.PLAYER_SIZE[1] // 2 - 1,
                    1,
                    -1
                ]),
                jnp.array([
                    state.player_x + self.consts.PLAYER_SIZE[0],
                    state.player_y + self.consts.PLAYER_SIZE[1] // 2 - 1,
                    1,
                    1
                ]),
            ),
            state.bullet,
        )

        # check if the new positions are in bounds, else destroy
        new_bullet = jnp.where(
            new_bullet[0] < self.consts.MIN_BOUND[0] - 2,
            jnp.array([0, 0, 0, 0]),
            jnp.where(
                new_bullet[0] > self.consts.MAX_BOUND[0] + self.consts.BULLET_SIZE[0] + 2,
                jnp.array([0, 0, 0, 0]),
                new_bullet
            ),
        )

        # if a bullet, we move the bullet further
        new_bullet = jnp.where(
            state.bullet[2],
            jnp.array([
                new_bullet[0] + new_bullet[3] * self.consts.BULLET_SPEED, # bullet speed
                new_bullet[1],
                new_bullet[2],
                new_bullet[3]
            ]),
            new_bullet,
        )

        # if bullet not active reset
        new_bullet = jnp.where(
            new_bullet[2] == 0,
            jnp.zeros_like(new_bullet),
            new_bullet
        )

        return new_bullet
    
    @partial(jax.jit, static_argnums=(0,))
    def prize_step(self, state: TurmoilState) -> chex.Array :
        """
        spawn a prize
        """
        def spawn(state: TurmoilState, rng_in) :
            rng_rest, if_spawn, _, lane, direction = self.spawn_data(
                rng_in,
                state,
                jnp.take(
                    jnp.array(self.consts.PRIZE_SPAWN_PROBABILITY),
                    state.level - 1
                ),
                0,
            )

            new_prize = jnp.where(
                if_spawn,
                jnp.array([
                    lane,
                    jnp.where(
                        direction == 1,
                        self.consts.ENEMY_LEFT_SPAWN_X,
                        self.consts.ENEMY_RIGHT_SPAWN_X
                    ), # X
                    jnp.take(jnp.array(self.consts.HORIZONTAL_LANES), lane) + self.consts.Y_OFFSET_PRIZE, #Y
                    1, # active
                    self.consts.PRIZE_TO_BOOM_TIME, # boom_timer
                    direction
                ]),
                state.prize
            )

            return new_prize, rng_rest

        # spawn new prize
        new_prize, rng_rest = jax.lax.cond(
            jnp.logical_and(state.prize[3] == 0, jnp.logical_not(state.game_phase == 2)),
            lambda : spawn(state, state.rng_key),
            lambda : (state.prize, state.rng_key),
        )

        # reduce boom_timer
        new_prize = new_prize.at[4].set(
            jnp.where(
                new_prize[3],
                new_prize[4] - 1,
                new_prize[4]
            )
        )

        return new_prize, rng_rest

    @partial(jax.jit, static_argnums=(0,))
    def spawn_data(self, rng: chex.PRNGKey, state: TurmoilState, spawn_prob: float = 1, for_enemy = 1):
        """
        Returns (new_rng, if_spawn, enemy_type, lane, direction).
        spawn_prob [0,1] controls probability of spawning when slots are available.
        """
        enemy_active_mask = (state.enemy[:, 3] == 1)
        prize_lane_mask = (jnp.arange(7) == state.prize[0]) & (state.prize[3] == 1)

        triangle_hollow_lane_mask = (
            (jnp.arange(7) == state.spawn_triangle_hollow[2])
            & (state.spawn_triangle_hollow[0] == 1)
        )

        inactive_mask = jnp.logical_not(
            enemy_active_mask | prize_lane_mask | triangle_hollow_lane_mask
        )

        num_inactive = jnp.sum(inactive_mask)

        def no_spawn(rng_in):
            return (
                rng_in,
                jnp.array(0, jnp.int32),
                jnp.array(0, jnp.int32),
                jnp.array(0, jnp.int32),
                jnp.array(0, jnp.int32),
            )

        def spawn(rng_in):
            rng_rest, type_key, lane_key, dir_key = jax.random.split(rng_in, 4)

            inactive_indices = jnp.nonzero(inactive_mask, size=state.enemy.shape[0])[0]
            lane_idx = jax.random.randint(lane_key, (), 0, num_inactive)
            lane = jax.lax.dynamic_index_in_dim(inactive_indices, lane_idx, keepdims=False)

            enemy_type = jax.random.choice(type_key, jnp.array([0, 1, 2, 3, 4, 5, 6, 7]))
            direction = jax.random.choice(dir_key, jnp.array([-1, 1]))

            return (
                rng_rest,
                jnp.array(1, jnp.int32),
                enemy_type.astype(jnp.int32),
                lane.astype(jnp.int32),
                direction.astype(jnp.int32),
            )

        def maybe_spawn(rng_in):
            rng_in, gate_key = jax.random.split(rng_in)
            do_spawn = jax.random.bernoulli(gate_key, p=spawn_prob)
            return jax.lax.cond(do_spawn, spawn, no_spawn, rng_in)
        
        def spawn_triangle_hollow_after_prize() :
            return (
                rng,
                jnp.array(1, jnp.int32),
                jnp.array(6, jnp.int32),
                jnp.array(state.spawn_triangle_hollow[2], jnp.int32),
                jnp.array(state.spawn_triangle_hollow[3], jnp.int32),
            )


        return jax.lax.cond(
            jnp.logical_and(
                for_enemy == 1,
                jnp.logical_and( # should spawn trinagle hollow in pre-defined lane
                    state.spawn_triangle_hollow[0] == 1,
                    jnp.mod(state.step_counter - state.spawn_triangle_hollow[1],
                            self.consts.STEP_COUNTER) >= self.consts.PRIZE_TILL_TRIANGLE_HOLLOW_TIMER
                )
            ),
            spawn_triangle_hollow_after_prize,
            lambda: jax.lax.cond(
                num_inactive == 0,
                no_spawn,
                maybe_spawn,
                rng
            ),
        )



    @partial(jax.jit, static_argnums=(0,))
    def change_type(self, state: TurmoilState):
        """
        Change type of arrow -> tank or pize -> sonic_boom
        """

        def change_arrow(enemy) :
            """
            change arrow to tank if timer has run out
            """
            arrow_lanes = enemy[:, 0] == 1
            arrow_reached_change_coordinate = enemy[:, 1] == enemy[:, 6]
            change_to_tank = arrow_lanes & arrow_reached_change_coordinate

            # new type
            new_type = jnp.where(
                change_to_tank,
                2,
                enemy[:, 0]
            )
            
            # new direction
            new_direction = jnp.where(
                change_to_tank,
                jnp.where(
                    enemy[:, 5] == 1,
                    -1,
                    1
                ),
                enemy[:, 5]
            )

            # new change_type_coordinate
            new_change_type_coordinate = jnp.where(
                new_direction == 1,
                self.consts.ENEMY_RIGHT_SPAWN_X,
                self.consts.ENEMY_LEFT_SPAWN_X
            )

            # update x
            new_y = jnp.where(
                change_to_tank, # remove arrow offset and add tank offset
                enemy[:, 2] - jnp.take(jnp.array(self.consts.Y_OFFSET_ENEMY), 1) + jnp.take(jnp.array(self.consts.Y_OFFSET_ENEMY), 2),
                enemy[:, 2]
            )

            # update enemy
            new_enemy = enemy.at[:, 0].set(new_type)
            new_enemy = new_enemy.at[:, 5].set(new_direction)
            new_enemy = new_enemy.at[:, 6].set(new_change_type_coordinate)
            new_enemy = new_enemy.at[:, 2].set(new_y)

            return new_enemy

        def change_prize(enemy, prize) :
            """
            change prize to sonic boom if timer has run out
            """
            # change active
            new_prize = prize.at[3].set(
                jnp.where(
                    jnp.logical_and(state.prize[3], state.prize[4] == 0),
                    0,
                    state.prize[3]
                )
            )
            
            # spawn sonic boom
            new_enemy = jnp.where(
                jnp.logical_and(state.prize[3], state.prize[4] == 0),
                enemy.at[state.prize[0].astype(jnp.int32)].set(
                    jnp.array([
                        8, # type sonic boom
                        state.prize[1], # x
                        state.prize[2] - self.consts.Y_OFFSET_PRIZE + jnp.take(jnp.array(self.consts.Y_OFFSET_ENEMY), 8), # y
                        1, # active
                        jnp.take(
                            jnp.take(jnp.array(self.consts.ENEMY_SPEED), 8, axis=0),
                            state.level - 1
                        ), # speed
                        state.prize[5], # direction
                        jnp.where( # change_type_coordiante
                            state.prize[5] == 1,
                            self.consts.ENEMY_RIGHT_SPAWN_X,
                            self.consts.ENEMY_LEFT_SPAWN_X
                        ),
                    ])
                ),
                enemy
            )

            return new_enemy, new_prize
        
        new_enemy = change_arrow(state.enemy)
        new_enemy, new_prize = change_prize(new_enemy, state.prize)

        return new_enemy, new_prize


    @partial(jax.jit, static_argnums=(0,))
    def enemy_spawn_step(self, state: TurmoilState) -> chex.Array :
        """
        Checks if it should spawn an enemy, if yes, spawns one.
        Decides type, position, direction and speed of enemy.
        """

        def spawn_fn(in_state, enemy_type, lane, direction) -> chex.Array :
            """
            Spawns enemy with given data for given state.

            Args:
               state, enemy_type, lane, direction
            """
            return in_state.enemy.at[lane].set(
                jnp.array([
                    enemy_type,
                    jnp.where(
                        direction == 1,
                        self.consts.ENEMY_LEFT_SPAWN_X,
                        self.consts.ENEMY_RIGHT_SPAWN_X
                    ), # X
                    jnp.take(jnp.array(self.consts.HORIZONTAL_LANES), lane) + jnp.take(jnp.array(self.consts.Y_OFFSET_ENEMY), enemy_type), #Y
                    1, # active
                    jnp.take(
                        jnp.take(jnp.array(self.consts.ENEMY_SPEED), enemy_type, axis=0),
                        state.level - 1
                    ), # speed
                    direction,
                    jnp.where( # change_type_coordiante
                        direction == 1,
                        self.consts.ENEMY_RIGHT_SPAWN_X,
                        self.consts.ENEMY_LEFT_SPAWN_X
                    ),
                ])
            )

        # get spawn data           
        rng_rest, if_spawn, enemy_type, lane, direction = self.spawn_data(
            state.rng_key,
            state,
            jnp.take(
                jnp.array(self.consts.ENEMY_SPAWN_PROBABILITY),
                state.level - 1
            ),
            1,
        )

        # spawn
        new_enemy = jax.lax.cond(
            if_spawn,
            lambda : spawn_fn(state, enemy_type, lane, direction),
            lambda : state.enemy,
        )

        # disable spawn_triangle_hollow
        should_spawn_triangle_hollow = jnp.logical_and(
            state.spawn_triangle_hollow[0] == 1,
            jnp.mod(state.step_counter - state.spawn_triangle_hollow[1],
                    self.consts.STEP_COUNTER) >= self.consts.PRIZE_TILL_TRIANGLE_HOLLOW_TIMER
        )

        new_spawn_triangle_hollow = jnp.where(
            should_spawn_triangle_hollow,
            jnp.zeros_like(state.spawn_triangle_hollow),
            state.spawn_triangle_hollow
        )

        return rng_rest, new_enemy, new_spawn_triangle_hollow


    @partial(jax.jit, static_argnums=(0,))
    def enemy_step(self, state: TurmoilState) :
        # move enemy
        new_enemy = state.enemy.at[:, 1].set(
            jnp.where(
                state.enemy[:, 3] == 1,
                state.enemy[:, 1] + state.enemy[:, 4] * state.enemy[:, 5],
                state.enemy[:, 1]
            )
        )

        # if tank or sonic_boom flip them back
        flip = jnp.logical_and(
            jnp.logical_or(
                new_enemy[:, 0] == 2, # type check
                new_enemy[:, 0] == 8
            ),
            (new_enemy[:, 1] - new_enemy[:, 6]) * new_enemy[:, 5] >= 0 # reached change type
        )

        ## set new direction
        new_enemy = new_enemy.at[:, 5].set(
            jnp.where(
                flip,
                -1 * new_enemy[:, 5],
                new_enemy[:, 5]
            )
        )

        ## set new change_type_coordinate
        new_enemy = new_enemy.at[:, 6].set(
            jnp.where(
                new_enemy[:, 5] == 1,
                self.consts.ENEMY_RIGHT_SPAWN_X,
                self.consts.ENEMY_LEFT_SPAWN_X
            )
        )

        # deactivate if out of bounds
        new_enemy = new_enemy.at[:, 3].set(
            jnp.where(
                jnp.logical_or(
                    new_enemy[:, 1] > self.consts.MAX_BOUND[0] + self.consts.ENEMY_SIZE_FOR_COLLISION[0],
                    new_enemy[:, 1] < self.consts.MIN_BOUND[0]
                ),
                0,
                new_enemy[:, 3]
            )
        )

        return new_enemy

    @partial(jax.jit, static_argnums=(0,))
    def update_score(self, state: TurmoilState, enemy_type) :
        new_score = jax.lax.switch(
            enemy_type,
            [
                lambda : state.score + 10,  # lines
                lambda : state.score + 100, # arrow
                lambda : state.score + 50,  # tank
                lambda : state.score + 40,  # L
                lambda : state.score + 60,  # T
                lambda : state.score + 30,  # rocket
                lambda : state.score,       # triangle_hollow
                lambda : state.score + 20,  # x_shape
                lambda : state.score + 100, # sonic_boom
            ]
        )

        return new_score
    
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

        return collisions
    
    
    @partial(jax.jit, static_argnums=(0,))
    def bullet_enemy_collision_step(self, state: TurmoilState):
        """
        Find collision of bullets with enemies and deactivate both
        in case of collision.
        Special rules:
            - triangle_hollow: no collision
            - tank: if same direction as bullet (shot from back) -> normal collision
                    if opposite direction as bullet (shot from front) -> bullet destroyed, enemy pushed back
        """
        bullet_pos = jnp.array([state.bullet[0], state.bullet[1]])
        bullet_size = self.consts.BULLET_SIZE

        enemy_pos = state.enemy[:, 1:3]
        enemy_active = state.enemy[:, 3]
        enemy_type = state.enemy[:, 0].astype(jnp.int32)
        enemy_dir = state.enemy[:, 5]
        enemy_size = self.consts.ENEMY_SIZE_FOR_COLLISION

        # mask inactive enemies
        active_enemy_pos = jnp.where(enemy_active[:, None] == 1, enemy_pos, -9999)

        # batch collision
        hit = self.check_collision_batch(bullet_pos, bullet_size, active_enemy_pos, enemy_size)

        # index of first hit
        hit_idx = jnp.argmax(jnp.where(hit, 1, 0))
        hit_type = jnp.where(jnp.any(hit), enemy_type[hit_idx], 0)


        def handle_triangle_hollow():
            # ignore collision, keep state
            return state.bullet, state.enemy, state.score

        def handle_tank():
            same_direction = enemy_dir[hit_idx] == state.bullet[3]

            def same_case():
                # full collision like normal
                new_enemy = state.enemy.at[hit_idx, 3].set(0)
                new_bullet = state.bullet.at[2].set(0)
                new_score = self.update_score(state, hit_type)
                return new_bullet, new_enemy, new_score

            def opposite_case():
                # bullet destroyed, enemy pushed back
                new_bullet = state.bullet.at[2].set(0)
                new_enemy = state.enemy.at[hit_idx, 1].add(
                    self.consts.TANK_PUSH_BACK * state.bullet[3]
                )
                return new_bullet, new_enemy, state.score

            return jax.lax.cond(same_direction, same_case, opposite_case)

        def handle_normal():
            new_enemy = jnp.where(
                state.bullet[2] == 1,
                state.enemy.at[:, 3].set(jnp.where(hit, 0, enemy_active)),
                state.enemy
            )
            new_bullet = jnp.where(
                state.bullet[2] == 1,
                state.bullet.at[2].set(jnp.where(jnp.any(hit), 0, state.bullet[2])),
                state.bullet
            )
            new_score = jax.lax.cond(
                jnp.any(hit),
                lambda: self.update_score(state, hit_type),
                lambda: state.score,
            )
            return new_bullet, new_enemy, new_score

        def handle_no_hit():
            return state.bullet, state.enemy, state.score

        return jax.lax.switch(
            jnp.where(jnp.any(hit), hit_type, 9),
            [
                handle_normal,  # lines
                handle_normal,  # arrow
                handle_tank,    # tank
                handle_normal,  # L
                handle_normal,  # T
                handle_normal,  # rocket
                handle_triangle_hollow,   # triangle_hollow
                handle_normal,  # x_shape
                handle_normal,  # sonic boom
                handle_no_hit,  # no hit case
            ],
        )


    @partial(jax.jit, static_argnums=(0,))
    def check_player_enemy_collision(self, state: TurmoilState) :
        # player rectangle
        player_pos = jnp.array([state.player_x, state.player_y])
        player_size = self.consts.PLAYER_SIZE

        enemy_pos = state.enemy[:, 1:3]   # take x, y columns
        enemy_active = state.enemy[:, 3]  # active flag
        enemy_size = self.consts.ENEMY_SIZE_FOR_COLLISION

        # mask inactive enemies
        active_enemy_pos = jnp.where(enemy_active[:, None] == 1, enemy_pos, -9999)

        # check collisions
        collisions = self.check_collision_batch(player_pos, player_size, active_enemy_pos, enemy_size)

        any_collision = jnp.any(collisions)

        # mark collided enemy inactive
        new_enemy_active = jnp.where(
            collisions,
            0,
            enemy_active
        )

        new_enemy = state.enemy.at[:, 3].set(new_enemy_active)

        new_ships = state.ships - any_collision.astype(jnp.int32)

        new_player_shrink = jnp.where(
            any_collision,
            1,
            state.player_shrink
        )

        return new_player_shrink, new_ships, new_enemy


    @partial(jax.jit, static_argnums=(0,))
    def prize_player_collision_step(self, state: TurmoilState):
        """
        Check player collision with prize
        """
        player_pos = jnp.array([state.player_x, state.player_y])
        
        # check collision
        collision = self.check_collision_single(
            player_pos,
            self.consts.PLAYER_SIZE, 
            jnp.array([state.prize[1], state.prize[2]]),
            self.consts.PRIZE_SIZE,
        )

        new_prize = jnp.where(
            collision,
            jnp.zeros_like(state.prize),
            state.prize,
        )

        new_score = jnp.where(
            collision,
            state.score + self.consts.PRIZE_SCORE_REWARD,
            state.score
        )

        # set spawn_triangle_hollow
        new_spawn_triangle_hollow = jnp.where(
            collision,
            jnp.array([1, state.step_counter, state.prize[0], state.prize[5] * -1]),
            state.spawn_triangle_hollow
        )

        return new_prize, new_score, new_spawn_triangle_hollow

    @partial(jax.jit, static_argnums=(0,))
    def game_control(self, state: TurmoilState) :
        """
        controls game phases
        """
        new_game_phase, new_game_phase_timer = state.game_phase, state.game_phase_timer + 1
        new_player_shrink = state.player_shrink

        def stay():
            return new_game_phase, new_game_phase_timer, new_player_shrink

        def to_loading():
            return (
                jnp.array(0),
                jnp.array(0),
                jnp.array(0),
            )

        def to_game():
            return (
                jnp.array(1),
                jnp.array(0),
                jnp.array(0),
            )
        
        def to_shrink_animation() :
            return (
                jnp.array(2),
                jnp.array(0),
                jnp.array(0),
            )
        
        return jax.lax.switch(
            new_game_phase,
            [
                # 0: show current lvl and colorful background 
                lambda: jax.lax.cond(
                    new_game_phase_timer >= self.consts.LOADING_GAME_PHASE_TIME,
                    to_game,
                    stay
                ),
                # 1: game
                lambda: jax.lax.cond(
                    state.player_shrink == 1,
                    lambda : to_shrink_animation(),
                    lambda : jax.lax.cond(
                        state.score >= jnp.take(jnp.array(self.consts.LVL_CHANGE_SCORES), state.level - 1),
                        lambda : to_loading(),
                        lambda : stay(),
                    ),
                ),
                # 2: animation for player shrinking
                lambda: jax.lax.cond(
                    new_game_phase_timer >= self.consts.PLAYER_SHRINK_TIME, 
                    to_game,
                    stay
                ),
            ],
        )
    

    @partial(jax.jit, static_argnums=(0,))
    def next_level(self, state: TurmoilState) :
        """
        move to next level
        """
        level_complete = state.score >= jnp.take(jnp.array(self.consts.LVL_CHANGE_SCORES), state.level - 1)

        # if bg is visible next level
        rng_rest, bg_key = jax.random.split(state.rng_key)
        next_bg_visible = jax.lax.cond(
            state.level + 1 < 4,
            lambda : 1,
            lambda : jax.random.bernoulli(bg_key, p=self.consts.BG_APPER_PROBABILITY).astype(jnp.int32)
        )

        next_state = state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_POS[0]),
            player_y=jnp.array(self.consts.PLAYER_START_POS[1]),
            player_direction=jnp.array(0),
            player_step_cooldown=jnp.zeros(2),
            player_move_unlock=jnp.zeros(2),
            player_shrink=jnp.array(0),
            ships=jnp.where(
                state.ships + 1 <= 7,
                state.ships + 1,
                state.ships
            ),
            score=state.score,

            bullet=jnp.zeros(4),

            enemy=jnp.zeros((7, 7)),
            prize=jnp.zeros(6),
            spawn_triangle_hollow=jnp.zeros(4),

            game_phase=state.game_phase,
            game_phase_timer=state.game_phase_timer,
            level=state.level + 1,
            step_counter=jnp.array(0),
            rng_key=rng_rest,
            bg_visible=next_bg_visible,
        )

        new_state = jax.lax.cond(
            level_complete,
            lambda: next_state,
            lambda: state
        )

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def _normal_game_step(self, state: TurmoilState, action: chex.Array) :
        """
        normal game step for playing
        """
        # player movement
        new_player_x, new_player_y, new_player_direction, new_player_step_cooldown, new_player_move_unlock= self.player_step(
            state,
            action
        )

        new_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_direction=new_player_direction,
            player_step_cooldown=new_player_step_cooldown,
            player_move_unlock=new_player_move_unlock,
        )

        # player collision
        new_player_shrink, new_ships, new_enemy = self.check_player_enemy_collision(
            new_state
        )

        new_state = new_state._replace(
            player_shrink=new_player_shrink,
            ships=new_ships, 
            enemy=new_enemy,
        )

        # bullet
        new_bullet = self.bullet_step(
            new_state,
            action
        )

        new_state = new_state._replace(
            bullet=new_bullet
        )

        # bullet enemy collision
        new_bullet, new_enemy, new_score = self.bullet_enemy_collision_step(
            new_state
        )
        
        new_state = new_state._replace(
            bullet=new_bullet,
            enemy=new_enemy,
            score=new_score
        )
        
        # enemy
        new_enemy = self.enemy_step(
            new_state,
        )

        new_state = new_state._replace(
            enemy=new_enemy
        )

        # spawn enemies
        new_rng, new_enemy, new_spawn_triangle_hollow = self.enemy_spawn_step(
            new_state
        )

        new_state = new_state._replace(
            enemy=new_enemy,
            rng_key=new_rng,
            spawn_triangle_hollow=new_spawn_triangle_hollow,
        )

        # change types
        new_enemy, new_prize = self.change_type(new_state)
        
        new_state = new_state._replace(
            enemy=new_enemy,
            prize=new_prize
        )

        # increment step_counter
        new_step_counter = jnp.where(
            new_state.step_counter == self.consts.STEP_COUNTER,
            jnp.array(0),
            new_state.step_counter + 1,
        )

        new_state = new_state._replace(
            step_counter=new_step_counter
        )
        
        # prize
        new_prize, rng_rest = self.prize_step(
            new_state,
        )

        new_state = new_state._replace(
            prize=new_prize,
            rng_key=rng_rest
        )

        # prize collision
        new_prize, new_score, new_spawn_triangle_hollow = self.prize_player_collision_step(new_state)

        new_state = new_state._replace(
            prize=new_prize,
            score=new_score,
            spawn_triangle_hollow=new_spawn_triangle_hollow,
        )

        # game control
        new_game_phase, new_game_phase_timer, new_player_shrink = self.game_control(
            new_state,
        )

        new_state = new_state._replace(
            game_phase=new_game_phase,
            game_phase_timer=new_game_phase_timer,
            player_shrink=new_player_shrink,
        )

        # go to next level
        new_state = self.next_level(new_state)
        
        return new_state


    @partial(jax.jit, static_argnums=(0,))
    def _game_loading_step(self, state: TurmoilState, action: chex.Array) :
        """
        step for loading animation at the start of levels
        """
        # increment step_counter
        new_step_counter = jnp.where(
            state.step_counter == self.consts.STEP_COUNTER,
            jnp.array(0),
            state.step_counter + 1,
        )

        new_state = state._replace(
            step_counter=new_step_counter
        )

        # game control
        new_game_phase, new_game_phase_timer, new_player_shrink = self.game_control(
            new_state,
        )

        new_state = new_state._replace(
            game_phase=new_game_phase,
            game_phase_timer=new_game_phase_timer,
            player_shrink=new_player_shrink,
        )

        return new_state
    
    @partial(jax.jit, static_argnums=(0,))
    def _player_shrink_animation_step(self, state: TurmoilState, action: chex.Array) :
        """
        step for shrinking player, if player enemy collision happens
        """
        # increment step_counter
        new_step_counter = jnp.where(
            state.step_counter == self.consts.STEP_COUNTER,
            jnp.array(0),
            state.step_counter + 1,
        )

        new_state = state._replace(
            step_counter=new_step_counter
        )

        # game control
        new_game_phase, new_game_phase_timer, new_player_shrink = self.game_control(
            new_state,
        )

        new_state = new_state._replace(
            game_phase=new_game_phase,
            game_phase_timer=new_game_phase_timer,
            player_shrink=new_player_shrink,
        )

        # bullet
        new_bullet = self.bullet_step(
            new_state,
            action
        )

        new_state = new_state._replace(
            bullet=new_bullet
        )

        # bullet enemy collision
        new_bullet, new_enemy, new_score = self.bullet_enemy_collision_step(
            new_state
        )
        
        new_state = new_state._replace(
            bullet=new_bullet,
            enemy=new_enemy,
            score=new_score
        )
        
        # enemy
        new_enemy = self.enemy_step(
            new_state,
        )

        new_state = new_state._replace(
            enemy=new_enemy
        )

        # change types
        new_enemy, new_prize = self.change_type(new_state)
        
        new_state = new_state._replace(
            enemy=new_enemy,
            prize=new_prize
        )

        # prize
        new_prize, rng_rest = self.prize_step(
            new_state,
        )

        new_state = new_state._replace(
            prize=new_prize,
            rng_key=rng_rest
        )

        return new_state
    
    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: TurmoilState, action: chex.Array
    ) -> Tuple[TurmoilObservation, TurmoilState, float, bool, TurmoilInfo]:

        previous_state = state

        return_state = jax.lax.switch(
            state.game_phase,
            [
                lambda s: self._game_loading_step(s, action),
                lambda s: self._normal_game_step(s, action),
                lambda s: self._player_shrink_animation_step(s, action),
            ],
            state,
        )

        observation = self._get_observation(return_state)
        done = self._get_done(return_state)
        env_reward = self._get_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

        return observation, return_state, env_reward, done, info

class TurmoilRenderer(JAXGameRenderer):
    def __init__(self, consts: TurmoilConstants = None):
        super().__init__()
        self.consts = consts or TurmoilConstants()
        self.player_shrink_offsets = len(PLAYER_SHRINK_OFFSETS)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TurmoilState):
        raster = jr.create_initial_frame(width=160, height=210)

        def _render_normal_game_step(raster) :
            # render background
            frame_bg = jr.get_sprite_frame(SPRITE_BG, 0)
            raster = jax.lax.cond(
                state.bg_visible,
                lambda r: jr.render_at(r, 0, 0, frame_bg),
                lambda r: r,
                raster
            )

            # render player
            def _render_player(raster):
                flip = state.player_direction == self.consts.FACE_LEFT

                def render_normal():
                    frame = jr.get_sprite_frame(PLAYER_SHIP, 0)
                    return jr.render_at(raster, state.player_x, state.player_y, frame, flip_horizontal=flip)

                def render_shrink():
                    frame = jr.get_sprite_frame(PLAYER_SHRINK, state.game_phase_timer)
                    return jr.render_at(raster, state.player_x, state.player_y, frame, flip_horizontal=flip)

                raster = jax.lax.cond(
                    state.game_phase == 2,
                    render_shrink,
                    render_normal,
                )

                return raster

            raster = _render_player(raster)

            # render bullet
            frame_bullet = jr.get_sprite_frame(BULLET, 0)
            raster = jax.lax.cond(
                state.bullet[2],
                lambda r: jr.render_at(
                    r,
                    state.bullet[0],
                    state.bullet[1],
                    frame_bullet
                ),
                lambda r: r,
                raster
            )

            # show the score
            score_array = jr.int_to_digits(state.score, max_digits=5)
            raster = jr.render_label(raster, 60, 10, score_array, DIGITS, spacing=8)

            # show remaining ships
            frame_pl_ship = jr.get_sprite_frame(PLAYER_SHIP, 0)
            raster = jnp.where(
                state.ships > 0,
                jr.render_indicator(
                    raster,
                    55 + (self.consts.PLAYER_SIZE[0]) * (5 - state.ships),
                    190,
                    state.ships - 1,
                    frame_pl_ship,
                    spacing=15
                ),
                raster
            )

            def _render_enemy(raster) :
                # render enemeis
                          
                frame_enemies, _ = jr.pad_to_match([
                    jr.get_sprite_frame(LINES_ENEMY, 0),
                    jr.get_sprite_frame(ARROW_ENEMY, 0),
                    jr.get_sprite_frame(TANK_ENEMY, state.step_counter),
                    jr.get_sprite_frame(L_ENEMY, state.step_counter),
                    jr.get_sprite_frame(T_ENEMY, state.step_counter),
                    jr.get_sprite_frame(ROCKET_ENEMY, state.step_counter),
                    jr.get_sprite_frame(TRIANGLE_HOLLOW_ENEMY, 0),
                    jr.get_sprite_frame(X_SHAPE_ENEMY, state.step_counter),
                    jr.get_sprite_frame(BOOM_ENEMY, state.step_counter),
                ])

                def render_enemy(i, r) :
                    return jax.lax.cond(
                        state.enemy[i, 3] == 1,
                        lambda r: jr.render_at(
                            r,
                            state.enemy[i, 1],
                            state.enemy[i, 2],
                            jnp.array(frame_enemies)[state.enemy[i, 0].astype(jnp.int32)],
                            flip_horizontal = state.enemy[i, 5] == self.consts.FACE_LEFT,
                        ),
                        lambda r: r,
                        r
                    )

                r = jax.lax.fori_loop(0, state.enemy.shape[0], render_enemy, raster)

                return r
            
            raster = _render_enemy(raster)

            # render prize
            frame_prize = jr.get_sprite_frame(PRIZE, state.step_counter)

            raster = jax.lax.cond(
                state.prize[3],
                lambda r: jr.render_at(
                    r,
                    state.prize[1],
                    state.prize[2],
                    frame_prize
                ),
                lambda r: r,
                raster
            )

            return raster


        def _render_loading_game_step(raster) :
            # show  level
            level_array = jr.int_to_digits(state.level, max_digits=1)
            raster = jr.render_label(raster, 76, 101, level_array, DIGITS, spacing=8)

            # show the score
            score_array = jr.int_to_digits(state.score, max_digits=5)
            raster = jr.render_label(raster, 60, 10, score_array, DIGITS, spacing=8)

            # show remaining ships
            frame_pl_ship = jr.get_sprite_frame(PLAYER_SHIP, 0)
            raster = jnp.where(
                state.ships > 0,
                jr.render_indicator(
                    raster,
                    55 + (self.consts.PLAYER_SIZE[0]) * (5 - state.ships),
                    190,
                    state.ships - 1,
                    frame_pl_ship,
                    spacing=15
                ),
                raster
            )

            return raster
        
        raster = jax.lax.switch(
            state.game_phase,
            [
                _render_loading_game_step,
                _render_normal_game_step,
                _render_normal_game_step,
            ],
            raster
        )

        return raster
