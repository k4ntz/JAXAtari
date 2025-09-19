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
    HORIZONTAL_LANES = [40, 61, 82, 103, 124, 145, 166] # y values

    # player
    PLAYER_SPEED = 10
    PLAYER_START_POS = (VERTICAL_LANE, HORIZONTAL_LANES[3]) # (starting_x_pos, starting_y_pos)
    PLAYER_STEP_COOLDOWN = (0, 20) # (x cooldown, y cooldown)
    PLAYER_STEP = (1, 21) # (x_step_size, y_step_size)

    # sizes
    PLAYER_SIZE = (8, 11) # (width, height)
    BULLET_SIZE = (8, 3)
    ENEMY_SIZE = (8, 8)

    # directions
    FACE_LEFT = -1
    FACE_RIGHT = 1

    # boundaries
    MIN_BOUND = (2, HORIZONTAL_LANES[0]) # (min x, min y)
    MAX_BOUND = (150, HORIZONTAL_LANES[-1])

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

class SpawnState(NamedTuple):
    pass

# Game state container
class TurmoilState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_step_cooldown: chex.Array # (2,) x_cooldown, y_cooldown
    ships: chex.Array
    score: chex.Array

    bullet: chex.Array # x, y, active, direction

    enemy: chex.Array # (7, 6) 7 lanes; 6 -> type (see constants), x, y, active, speed, direction
    enemy_spawn_timer: chex.Array # delay spawning

    step_counter: chex.Array
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


class TurmoilObservation(NamedTuple):
    player: PlayerEntity
    ships: jnp.array

class TurmoilInfo(NamedTuple):
    step_counter: jnp.ndarray  # Current step count
    all_rewards: jnp.ndarray  # All rewards for the current step



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

    # maybe change this later with render part as well
    lines_enemy_sprites = [lines_enemy]
    arrow_enemy_sprites = [arrow_enemy]
    boom_enemy_sprites, _ = jr.pad_to_match([boom_enemy_1, boom_enemy_2])
    L_enemy_sprites, _ = jr.pad_to_match([L_enemy_1, L_enemy_2])
    rocket_enemy_sprites, _ = jr.pad_to_match([rocket_enemy_1, rocket_enemy_2])
    T_enemy_sprites, _ = jr.pad_to_match([T_enemy_1, T_enemy_2])
    tank_enemy_sprites, _ = jr.pad_to_match([tank_enemy_1, tank_enemy_2])
    triangle_hollow_enemy_sprites = [triangle_hollow_enemy]
    x_shape_enemy_sprites, _ = jr.pad_to_match([x_shape_enemy_1, x_shape_enemy_2])
    
    # sprites_enemy, _ = jr.pad_to_match([lines_enemy, arrow_enemy, boom_enemy_1, boom_enemy_2, L_enemy_1, L_enemy_2,
    #                                     rocket_enemy_1, rocket_enemy_2, T_enemy_1, T_enemy_2, tank_enemy_1, tank_enemy_2, triangle_hollow_enemy, x_shape_enemy_1, x_shape_enemy_2])
    

    # lines_enemy_sprites = [sprites_enemy[0]]
    # arrow_enemy_sprites = [sprites_enemy[1]]
    # boom_enemy_sprites = [sprites_enemy[2], sprites_enemy[3]]
    # L_enemy_sprites = [sprites_enemy[4], sprites_enemy[5]]
    # rocket_enemy_sprites = [sprites_enemy[6], sprites_enemy[7]]
    # T_enemy_sprites = [sprites_enemy[8], sprites_enemy[9]]
    # tank_enemy_sprites = [sprites_enemy[10], sprites_enemy[11]]
    # triangle_hollow_enemy_sprites = [sprites_enemy[12]]
    # x_shape_enemy_sprites = [sprites_enemy[13], sprites_enemy[14]]

    # bg sprites
    SPRITE_BG = jnp.repeat(bg[0][None], 1, axis=0)

    # Player sprites
    PLAYER_SHIP = jnp.repeat(player[0][None], 1, axis=0)

    # bullet sprites
    BULLET = jnp.repeat(bullet[0][None], 1, axis=0)

    # player shrink sprites
    PLAYER_SHRINK = jnp.concatenate(
        [
            jnp.repeat(player_shrink_sprites[0][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[1][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[2][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[3][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[4][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[5][None], 4, axis=0),
        ]
    )

    DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/turmoil/digits/{}.npy"))

    # enemeis
    LINES_ENEMY = jnp.repeat(lines_enemy_sprites[0][None], 1, axis=0)

    ARROW_ENEMY = jnp.repeat(arrow_enemy_sprites[0][None], 1, axis=0)

    BOOM_ENEMY = jnp.concatenate(
        [
            jnp.repeat(boom_enemy_sprites[0][None], 16, axis=0),
            jnp.repeat(boom_enemy_sprites[1][None], 16, axis=0),
        ]
    )

    L_ENEMY = jnp.concatenate(
        [
            jnp.repeat(L_enemy_sprites[0][None], 16, axis=0),
            jnp.repeat(L_enemy_sprites[1][None], 16, axis=0),
        ]
    )

    ROCKET_ENEMY = jnp.concatenate(
        [
            jnp.repeat(rocket_enemy_sprites[0][None], 16, axis=0),
            jnp.repeat(rocket_enemy_sprites[1][None], 16, axis=0),
        ]
    )

    T_ENEMY = jnp.concatenate(
        [
            jnp.repeat(T_enemy_sprites[0][None], 16, axis=0),
            jnp.repeat(T_enemy_sprites[1][None], 16, axis=0),
        ]
    )

    TANK_ENEMY = jnp.concatenate(
        [
            jnp.repeat(tank_enemy_sprites[0][None], 16, axis=0),
            jnp.repeat(tank_enemy_sprites[1][None], 16, axis=0),
        ]
    )

    TRIANGLE_HOLLOW_ENEMY = jnp.repeat(triangle_hollow_enemy_sprites[0][None], 1, axis=0)

    X_SHAPE_ENEMY = jnp.concatenate(
        [
            jnp.repeat(x_shape_enemy_sprites[0][None], 16, axis=0),
            jnp.repeat(x_shape_enemy_sprites[1][None], 16, axis=0),
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
        self.obs_size = 6 + 12 * 5 + 12 * 5 + 4 * 5 + 4 * 5 + 5 + 5 + 4
        self.renderer = TurmoilRenderer(self.consts)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TurmoilState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: TurmoilState) -> TurmoilObservation:
        # Create player (already scalar, no need for vectorization)
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            o=state.player_direction,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
            active=jnp.array(1),  # Player is always active
        )

        return TurmoilObservation(
            player=player,
            ships=state.ships
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TurmoilState, all_rewards: jnp.ndarray) -> TurmoilInfo:
        return TurmoilInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: TurmoilState, state: TurmoilState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: TurmoilState, state: TurmoilState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TurmoilState) -> bool:
        return state.ships < 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[TurmoilObservation, TurmoilState]:
        """Initialize game state"""
        reset_state = TurmoilState(
            player_x=jnp.array(self.consts.PLAYER_START_POS[0]),
            player_y=jnp.array(self.consts.PLAYER_START_POS[1]),
            player_direction=jnp.array(0),
            player_step_cooldown=jnp.zeros(2),
            ships=jnp.array(5),
            score=jnp.array(0),

            bullet=jnp.zeros(4),

            enemy=jnp.zeros((7, 6)),
            enemy_spawn_timer=jnp.array(0),

            step_counter=jnp.array(0),
            rng_key=key,
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

        # cooldown so player does not go too fast
        player_step_cooldown = jnp.where(
            state.player_step_cooldown > 0,
            state.player_step_cooldown - 1,
            0
        )

        # move player
        player_x = jnp.where(
            player_step_cooldown[0] <= 0,
            jnp.where(
                right,
                state.player_x + self.consts.PLAYER_STEP[0],
                jnp.where(
                    left,
                    state.player_x - self.consts.PLAYER_STEP[0],
                    state.player_x
                )
            ),
            state.player_x
        )

        player_y = jnp.where(
            player_step_cooldown[1] <= 0,
            jnp.where(
                down,
                state.player_y + self.consts.PLAYER_STEP[1],
                jnp.where(
                    up,
                    state.player_y - self.consts.PLAYER_STEP[1],
                    state.player_y
                )
            ),
            state.player_y
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
            self.consts.MIN_BOUND[1],
            jnp.where(
                player_y >= self.consts.MAX_BOUND[1],
                self.consts.MAX_BOUND[1],
                player_y
            ),
        )

        return player_x, player_y, player_direction, player_step_cooldown


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
            jnp.logical_and(fire, jnp.logical_not(state.bullet[2])),
            jnp.where(
                state.player_direction == -1,
                jnp.array([
                    state.player_x - self.consts.BULLET_SIZE[0], # x, y, active, direction
                    state.player_y + self.consts.PLAYER_SIZE[1] / 2,
                    1,
                    -1
                ]),
                jnp.array([
                    state.player_x + self.consts.PLAYER_SIZE[0],
                    state.player_y + self.consts.PLAYER_SIZE[1] / 2,
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
                new_bullet[0] + new_bullet[3] * 3, # bullet speed
                new_bullet[1],
                new_bullet[2],
                new_bullet[3]
            ]),
            new_bullet,
        )

        return new_bullet

    @partial(jax.jit, static_argnums=(0,))
    def enemy_spawn_step(self, state: TurmoilState) -> chex.Array :
        """
        Checks if it should spawn an enemy, if yes, spawns one.
        Decides type, position, direction and speed of enemy.
        """
        
        def spawn_data(rng: chex.PRNGKey, in_state: TurmoilState):
            """
            Returns (new_rng, if_spawn, enemy_type, lane, speed, direction).
            """
            inactive_mask = in_state.enemy[:, 3] == 0
            num_inactive = jnp.sum(inactive_mask)

            def no_spawn(rng_in):
                return (
                    rng_in,
                    jnp.array(0, jnp.int32),
                    jnp.array(0, jnp.int32),
                    jnp.array(0, jnp.int32),
                    jnp.array(0.0, jnp.float32),
                    jnp.array(0, jnp.int32),
                )

            def do_spawn(rng_in):
                rng_rest, type_key, lane_key, speed_key, dir_key = jax.random.split(rng_in, 5)

                inactive_indices = jnp.nonzero(inactive_mask, size=in_state.enemy.shape[0])[0]
                lane_idx = jax.random.randint(lane_key, (), 0, num_inactive)
                lane = jax.lax.dynamic_index_in_dim(inactive_indices, lane_idx, keepdims=False)

                enemy_type = jax.random.randint(type_key, (), 0, 8)
                speed = jax.random.uniform(speed_key, (), minval=0.5, maxval=1.0)
                direction = jax.random.choice(dir_key, jnp.array([-1, 1]))

                return (
                    rng_rest,
                    jnp.array(1, jnp.int32),
                    enemy_type.astype(jnp.int32),
                    lane.astype(jnp.int32),
                    speed.astype(jnp.float32),
                    direction.astype(jnp.int32),
                )

            return jax.lax.cond(num_inactive == 0, no_spawn, do_spawn, rng)


        def spawn_fn(data) -> chex.Array :
            """
            Spawns enemy with given data for given state.

            Args:
                data : Tuple containing (state, enemy_type, lane, direction, speed)
            """
            (in_state, enemy_type, lane, direction, speed) = data

            return in_state.enemy.at[lane].set(
                jnp.array([
                    enemy_type,
                    jnp.where(
                        direction == 1,
                        10,
                        100
                    ), # X
                    jnp.take(jnp.array(self.consts.HORIZONTAL_LANES), lane), #Y
                    1, # active
                    speed,
                    direction
                ])
            )

        (   
            rng_rest,
            if_spawn,
            enemy_type,
            lane,
            speed,
            direction,
        ) = spawn_data(state.rng_key, state)

        new_enemy = jax.lax.cond(
            if_spawn,
            lambda data: spawn_fn(data),
            lambda _: state.enemy,
            (state, enemy_type, lane, direction, speed)
        )

        # spawn timer
        new_enemy = jnp.where(
            state.enemy_spawn_timer == 0,
            new_enemy,
            state.enemy
        )

        new_enemy_spawn_timer = jnp.where(
            state.enemy_spawn_timer == 0,
            20,
            state.enemy_spawn_timer - 1
        )

        return rng_rest, new_enemy, new_enemy_spawn_timer


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

        # deactivate if out of bounds
        new_enemy = new_enemy.at[:, 3].set(
            jnp.where(
                jnp.logical_or(
                    new_enemy[:, 1] > self.consts.MAX_BOUND[0], # TODO add enemy width for accuracy
                    new_enemy[:, 1] < self.consts.MIN_BOUND[0]
                ),
                0,
                new_enemy[:, 3]
            )
        )

        return new_enemy

    def update_score(self, state: TurmoilState, enemy_type) :
        new_score = jax.lax.switch(
            enemy_type,
            [
                lambda : state.score + 10, # lines
                lambda : state.score + 100, # arrow
                lambda : state.score + 50, # tank
                lambda : state.score + 40, # L
                lambda : state.score + 60, # T
                lambda : state.score + 30, # rocket
                lambda : state.score + 10, # triangle_hollow
                lambda : state.score + 20, # x_shape
                lambda : state.score + 100, # sonic_boom
            ]
        )

        return new_score

    def bullet_enemy_collision_step(self, state: TurmoilState):
        """
        Find collision of bullets with enemies and deactivate both
        in case of collision
        """
        bx, by = state.bullet[0], state.bullet[1]
        ex = state.enemy[:, 1]
        ey = state.enemy[:, 2]
        active = state.enemy[:, 3]

        w, h = self.consts.ENEMY_SIZE

        hit = (
            (active == 1) &
            (bx >= ex) & (bx <= ex + w) &
            (by >= ey) & (by <= ey + h)
        )

        # deactivate  collided enemy
        new_enemy = jnp.where(
            state.bullet[2] == 1,
            state.enemy.at[:, 3].set(jnp.where(hit, 0, active)),
            state.enemy
        )

        # deactivate bullet if collision
        new_bullet = jnp.where(
            state.bullet[2] == 1,
            state.bullet.at[2].set(jnp.where(jnp.any(hit), 0, state.bullet[2])),
            state.bullet
        )

        # if any enemy hit then update score (assumption only 1 enemy shot at a time)
        new_score = jax.lax.cond(
            jnp.any(hit),
            lambda : self.update_score(state, 0),
            lambda : state.score,
        )

        return new_bullet, new_enemy, new_score

    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: TurmoilState, action: chex.Array
    ) -> Tuple[TurmoilObservation, TurmoilState, float, bool, TurmoilInfo]:

        previous_state = state
        _, reset_state = self.reset()

        def normal_game_step() :
            # player movement
            new_player_x, new_player_y, new_player_direction, new_player_step_cooldown = self.player_step(
                state,
                action
            )

            new_state = state._replace(
                player_x=new_player_x,
                player_y=new_player_y,
                player_direction=new_player_direction,
                player_step_cooldown=new_player_step_cooldown
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
            new_rng, new_enemy, new_enemy_spawn_timer = self.enemy_spawn_step(
                new_state
            )

            new_state = new_state._replace(
                enemy=new_enemy,
                rng_key=new_rng,
                enemy_spawn_timer=new_enemy_spawn_timer,
            )

            # increment step_counter
            new_step_counter = jnp.where(
                new_state.step_counter == 1024,
                jnp.array(0),
                new_state.step_counter + 1,
            )

            new_state = new_state._replace(
                step_counter=new_step_counter
            )

            return new_state


        return_state = normal_game_step()

        observation = self._get_observation(return_state)
        done = self._get_done(return_state)
        env_reward = self._get_env_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

        return observation, return_state, env_reward, done, info

class TurmoilRenderer(JAXGameRenderer):
    def __init__(self, consts: TurmoilConstants = None):
        super().__init__()
        self.consts = consts or TurmoilConstants()
        self.player_shrink_offsets = len(PLAYER_SHRINK_OFFSETS)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        # render background
        frame_bg = jr.get_sprite_frame(SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        # render player
        frame_pl_ship = jr.get_sprite_frame(PLAYER_SHIP, 0)
        raster = jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_pl_ship,
            flip_horizontal = state.player_direction == self.consts.FACE_LEFT,
        )

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
        score_array = jr.int_to_digits(state.score, max_digits=4)
        raster = jr.render_label(raster, 65, 10, score_array, DIGITS, spacing=8)

        # show remaining ships
        frame_pl_ship = jr.get_sprite_frame(PLAYER_SHIP, 0)
        raster = jnp.where(
            state.ships - 1 >= 0,
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
            frame_3lines_enemy = jr.get_sprite_frame(LINES_ENEMY, 0)
            frame_arrow_enemy = jr.get_sprite_frame(ARROW_ENEMY, 0)
            frame_boom_enemy = jr.get_sprite_frame(BOOM_ENEMY, state.step_counter)
            frame_L_enemy = jr.get_sprite_frame(L_ENEMY, state.step_counter)
            frame_rocket_enemy = jr.get_sprite_frame(ROCKET_ENEMY, state.step_counter)
            frame_T_enemy = jr.get_sprite_frame(T_ENEMY, state.step_counter)
            frame_tank_enemy = jr.get_sprite_frame(TANK_ENEMY, state.step_counter)
            frame_triangle_hollow_enemy = jr.get_sprite_frame(TRIANGLE_HOLLOW_ENEMY, 0)
            frame_x_shape_enemy = jr.get_sprite_frame(X_SHAPE_ENEMY, state.step_counter)

            # maybe change this later with get_sprites part as well
            frame_enemies = [
                frame_3lines_enemy,
                frame_arrow_enemy,
                frame_boom_enemy,
                frame_L_enemy,
                frame_rocket_enemy,
                frame_T_enemy,
                frame_tank_enemy,
                frame_triangle_hollow_enemy,
                frame_x_shape_enemy
            ]

            # Utility to pad a sprite to target shape
            def pad_to_shape(arr: jnp.ndarray, target_shape: tuple[int, int, int]) -> jnp.ndarray:
                h, w, c = arr.shape
                H, W, C = target_shape
                out = jnp.zeros(target_shape, dtype=arr.dtype)
                out = out.at[:h, :w, :c].set(arr)
                return out

            # Determine max H, W, C
            max_h = max(f.shape[0] for f in frame_enemies)
            max_w = max(f.shape[1] for f in frame_enemies)
            max_c = max(f.shape[2] for f in frame_enemies)
            target_shape = (max_h, max_w, max_c)

            # Pad all frames to the same shape
            frame_enemies = [pad_to_shape(f, target_shape) for f in frame_enemies]

            # JAX-safe selection
            def get_enemy_frame(enemy_id: int) -> jnp.ndarray:
                return jax.lax.switch(
                    enemy_id,
                    [lambda f=f: f for f in frame_enemies],
                )
            
            # def get_enemy_frame(enemy_id: int):
            #     return jax.lax.switch(
            #         enemy_id,
            #         [
            #             lambda: frame_3lines_enemy,
            #             lambda: frame_arrow_enemy,
            #             lambda: frame_boom_enemy,
            #             lambda: frame_L_enemy,
            #             lambda: frame_rocket_enemy,
            #             lambda: frame_T_enemy,
            #             lambda: frame_tank_enemy,
            #             lambda: frame_triangle_hollow_enemy,
            #             lambda: frame_x_shape_enemy,
            #         ],
            #     )

            def render_enemy(i, r) :
                enemy_id = state.enemy[i, 0].astype(int)
                return jax.lax.cond(
                    state.enemy[i, 3] == 1,
                    lambda r: jr.render_at(
                        r,
                        state.enemy[i, 1],
                        state.enemy[i, 2],
                        get_enemy_frame(enemy_id),
                        flip_horizontal = state.enemy[i, 5] == self.consts.FACE_LEFT,
                    ),
                    lambda r: r,
                    r
                )

            r = jax.lax.fori_loop(0, state.enemy.shape[0], render_enemy, raster)

            return r
        
        raster = _render_enemy(raster)

        return raster
