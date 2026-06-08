from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class JamesBondConstants(struct.PyTreeNode):
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    GAME_AREA_MIN_X: int = struct.field(pytree_node=False, default=8)
    GAME_AREA_MAX_X: int = struct.field(pytree_node=False, default=152)
    GAME_AREA_MIN_Y: int = struct.field(pytree_node=False, default=28)
    GAME_AREA_MAX_Y: int = struct.field(pytree_node=False, default=196)

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=10)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=8)
    PLAYER_INIT_X: int = struct.field(pytree_node=False, default=32)
    PLAYER_INIT_Y: int = struct.field(pytree_node=False, default=160)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=2.0)
    GRAVITY: float = struct.field(pytree_node=False, default=0.0)
    JUMP_VELOCITY: float = struct.field(pytree_node=False, default=0.0)

    MAX_LIVES: int = struct.field(pytree_node=False, default=3)
    MAX_DIAMONDS: int = struct.field(pytree_node=False, default=8)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_BULLETS: int = struct.field(pytree_node=False, default=4)
    MAX_EPISODE_STEPS: int = struct.field(pytree_node=False, default=5000)

    DIAMOND_WIDTH: int = struct.field(pytree_node=False, default=4)
    DIAMOND_HEIGHT: int = struct.field(pytree_node=False, default=4)
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=10)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=8)
    BULLET_WIDTH: int = struct.field(pytree_node=False, default=3)
    BULLET_HEIGHT: int = struct.field(pytree_node=False, default=2)

    REWARD_STEP: float = struct.field(pytree_node=False, default=0.0)
    REWARD_DIAMOND: float = struct.field(pytree_node=False, default=1.0)
    REWARD_HIT_ENEMY: float = struct.field(pytree_node=False, default=-1.0)
    REWARD_LOST_LIFE: float = struct.field(pytree_node=False, default=-1.0)

    ACTION_MEANINGS: Tuple[str, ...] = struct.field(
        pytree_node=False,
        default=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
    )

    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(8, 14, 32)
    )
    PLAY_AREA_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(20, 42, 66)
    )
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(236, 236, 236)
    )
    DIAMOND_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(0, 216, 255)
    )
    ENEMY_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(220, 64, 64)
    )
    BULLET_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default=(250, 220, 72)
    )


@struct.dataclass
class JamesBondState:
    player_x: chex.Array
    player_y: chex.Array
    player_vx: chex.Array
    player_vy: chex.Array
    player_direction: chex.Array
    lives: chex.Array
    score: chex.Array
    step_count: chex.Array
    level_progress: chex.Array
    diamond_x: chex.Array
    diamond_y: chex.Array
    diamond_active: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_active: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_vx: chex.Array
    bullet_active: chex.Array
    collision_happened: chex.Array
    collected_diamond: chex.Array
    hit_enemy: chex.Array
    fired_bullet: chex.Array
    key: chex.PRNGKey


@struct.dataclass
class JamesBondObservation:
    player: ObjectObservation
    diamonds: ObjectObservation
    enemies: ObjectObservation
    bullets: ObjectObservation
    player_velocity: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray
    level_progress: jnp.ndarray


@struct.dataclass
class JamesBondInfo:
    collision_happened: jnp.ndarray
    collected_diamond: jnp.ndarray
    hit_enemy: jnp.ndarray
    fired_bullet: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    level_progress: jnp.ndarray
    step_count: jnp.ndarray


class JaxJamesBond(
    JaxEnvironment[JamesBondState, JamesBondObservation, JamesBondInfo, JamesBondConstants]
):
    # Compact agent action indices map to these ALE-style actions.
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.FIRE],
        dtype=jnp.int32,
    )

    def __init__(self, consts: JamesBondConstants = None):
        consts = consts or JamesBondConstants()
        super().__init__(consts)
        self.renderer = JamesBondRenderer(self.consts)

    def reset(
        self, key: chex.PRNGKey = jax.random.PRNGKey(0)
    ) -> Tuple[JamesBondObservation, JamesBondState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        state_key, _ = jax.random.split(key)

        state = JamesBondState(
            player_x=jnp.array(self.consts.PLAYER_INIT_X, dtype=jnp.float32),
            player_y=jnp.array(self.consts.PLAYER_INIT_Y, dtype=jnp.float32),
            player_vx=jnp.array(0.0, dtype=jnp.float32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            player_direction=jnp.array(1, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            level_progress=jnp.array(0, dtype=jnp.int32),
            diamond_x=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.float32),
            diamond_y=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.float32),
            diamond_active=jnp.zeros((self.consts.MAX_DIAMONDS,), dtype=jnp.bool_),
            enemy_x=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_y=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.float32),
            enemy_active=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.bool_),
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_vx=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.float32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), dtype=jnp.bool_),
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            fired_bullet=jnp.array(False, dtype=jnp.bool_),
            key=state_key,
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: JamesBondState, action: chex.Array
    ) -> Tuple[JamesBondObservation, JamesBondState, chex.Array, chex.Array, JamesBondInfo]:
        atari_action = self._decode_action(action)
        previous_state = state

        state = state.replace(
            step_count=state.step_count + 1,
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
            fired_bullet=atari_action == Action.FIRE,
        )
        state = self._update_player_placeholder(state, atari_action)
        state = self._update_objects_placeholder(state)
        state = self._check_collisions_placeholder(state)

        _, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key)

        observation = self._get_observation(state)
        reward = self._calculate_reward_placeholder(previous_state, state)
        done = self._is_done(state)
        info = self._get_info(state)

        return observation, state, reward, done, info

    def render(self, state: JamesBondState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        screen_size = (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)
        return spaces.Dict(
            {
                "player": spaces.get_object_space(n=None, screen_size=screen_size),
                "diamonds": spaces.get_object_space(
                    n=self.consts.MAX_DIAMONDS, screen_size=screen_size
                ),
                "enemies": spaces.get_object_space(
                    n=self.consts.MAX_ENEMIES, screen_size=screen_size
                ),
                "bullets": spaces.get_object_space(
                    n=self.consts.MAX_BULLETS, screen_size=screen_size
                ),
                "player_velocity": spaces.Box(
                    low=jnp.array([-10.0, -20.0], dtype=jnp.float32),
                    high=jnp.array([10.0, 20.0], dtype=jnp.float32),
                    shape=(2,),
                    dtype=jnp.float32,
                ),
                "lives": spaces.Box(
                    low=0,
                    high=self.consts.MAX_LIVES,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "score": spaces.Box(
                    low=0,
                    high=1_000_000,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "level_progress": spaces.Box(
                    low=0,
                    high=self.consts.MAX_EPISODE_STEPS,
                    shape=(),
                    dtype=jnp.int32,
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: JamesBondState) -> JamesBondObservation:
        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
            active=jnp.array(True, dtype=jnp.bool_),
            orientation=jnp.where(state.player_direction < 0, 270.0, 90.0),
        )
        diamonds = self._object_group_observation(
            state.diamond_x,
            state.diamond_y,
            state.diamond_active,
            self.consts.DIAMOND_WIDTH,
            self.consts.DIAMOND_HEIGHT,
        )
        enemies = self._object_group_observation(
            state.enemy_x,
            state.enemy_y,
            state.enemy_active,
            self.consts.ENEMY_WIDTH,
            self.consts.ENEMY_HEIGHT,
        )
        bullets = self._object_group_observation(
            state.bullet_x,
            state.bullet_y,
            state.bullet_active,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            orientation=jnp.where(state.bullet_vx < 0, 270.0, 90.0),
        )
        return JamesBondObservation(
            player=player,
            diamonds=diamonds,
            enemies=enemies,
            bullets=bullets,
            player_velocity=jnp.stack([state.player_vx, state.player_vy]).astype(
                jnp.float32
            ),
            lives=state.lives,
            score=state.score,
            level_progress=state.level_progress,
        )

    def _object_group_observation(
        self,
        x: chex.Array,
        y: chex.Array,
        active: chex.Array,
        width: int,
        height: int,
        orientation: chex.Array = None,
    ) -> ObjectObservation:
        return ObjectObservation.create(
            x=x,
            y=y,
            width=jnp.full(x.shape, width, dtype=jnp.int32),
            height=jnp.full(y.shape, height, dtype=jnp.int32),
            active=active,
            orientation=orientation,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: JamesBondState) -> JamesBondInfo:
        return JamesBondInfo(
            collision_happened=state.collision_happened,
            collected_diamond=state.collected_diamond,
            hit_enemy=state.hit_enemy,
            fired_bullet=state.fired_bullet,
            score=state.score,
            lives=state.lives,
            level_progress=state.level_progress,
            step_count=state.step_count,
        )

    def _decode_action(self, action: chex.Array) -> chex.Array:
        return jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

    def _update_player_placeholder(
        self, state: JamesBondState, atari_action: chex.Array
    ) -> JamesBondState:
        left = atari_action == Action.LEFT
        right = atari_action == Action.RIGHT
        up = atari_action == Action.UP
        down = atari_action == Action.DOWN

        player_vx = (
            right.astype(jnp.float32) - left.astype(jnp.float32)
        ) * self.consts.PLAYER_SPEED
        player_vy = (
            down.astype(jnp.float32) - up.astype(jnp.float32)
        ) * self.consts.PLAYER_SPEED

        player_x = jnp.clip(
            state.player_x + player_vx,
            self.consts.GAME_AREA_MIN_X,
            self.consts.GAME_AREA_MAX_X - self.consts.PLAYER_WIDTH,
        )
        player_y = jnp.clip(
            state.player_y + player_vy,
            self.consts.GAME_AREA_MIN_Y,
            self.consts.GAME_AREA_MAX_Y - self.consts.PLAYER_HEIGHT,
        )
        player_direction = jnp.where(
            left, -1, jnp.where(right, 1, state.player_direction)
        ).astype(jnp.int32)

        return state.replace(
            player_x=player_x.astype(jnp.float32),
            player_y=player_y.astype(jnp.float32),
            player_vx=player_vx.astype(jnp.float32),
            player_vy=player_vy.astype(jnp.float32),
            player_direction=player_direction,
        )

    def _update_objects_placeholder(self, state: JamesBondState) -> JamesBondState:
        # Future object lifecycle logic belongs here.
        return state

    def _check_collisions_placeholder(self, state: JamesBondState) -> JamesBondState:
        # Future diamond, enemy, bullet, and life collision logic belongs here.
        return state.replace(
            collision_happened=jnp.array(False, dtype=jnp.bool_),
            collected_diamond=jnp.array(False, dtype=jnp.bool_),
            hit_enemy=jnp.array(False, dtype=jnp.bool_),
        )

    def _calculate_reward_placeholder(
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        del previous_state, state
        return jnp.array(self.consts.REWARD_STEP, dtype=jnp.float32)

    def _is_done(self, state: JamesBondState) -> chex.Array:
        return jnp.logical_or(
            state.lives <= 0,
            state.step_count >= self.consts.MAX_EPISODE_STEPS,
        )


class JamesBondRenderer(JAXGameRenderer):
    def __init__(
        self,
        consts: JamesBondConstants = None,
        config: render_utils.RendererConfig = None,
    ):
        self.consts = consts or JamesBondConstants()
        if config is None:
            config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None,
            )
        super().__init__(self.consts, config)
        self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        self.PALETTE = jnp.array(
            [
                self.consts.BACKGROUND_COLOR,
                self.consts.PLAY_AREA_COLOR,
                self.consts.PLAYER_COLOR,
                self.consts.DIAMOND_COLOR,
                self.consts.ENEMY_COLOR,
                self.consts.BULLET_COLOR,
            ],
            dtype=jnp.uint8,
        )
        self.BACKGROUND_ID = 0
        self.PLAY_AREA_ID = 1
        self.PLAYER_ID = 2
        self.DIAMOND_ID = 3
        self.ENEMY_ID = 4
        self.BULLET_ID = 5
        self.BACKGROUND = jnp.full(
            (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
            self.BACKGROUND_ID,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: JamesBondState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_background(raster)
        raster = self._render_objects(raster, state)
        raster = self._render_player(raster, state)
        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_background(self, raster: jnp.ndarray) -> jnp.ndarray:
        position = jnp.array(
            [[self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MIN_Y]],
            dtype=jnp.int32,
        )
        size = jnp.array(
            [
                [
                    self.consts.GAME_AREA_MAX_X - self.consts.GAME_AREA_MIN_X,
                    self.consts.GAME_AREA_MAX_Y - self.consts.GAME_AREA_MIN_Y,
                ]
            ],
            dtype=jnp.int32,
        )
        return self.jr.draw_rects(raster, position, size, self.PLAY_AREA_ID)

    def _render_player(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        position = jnp.stack(
            [
                jnp.round(state.player_x).astype(jnp.int32),
                jnp.round(state.player_y).astype(jnp.int32),
            ]
        )[None, :]
        size = jnp.array(
            [[self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT]], dtype=jnp.int32
        )
        return self.jr.draw_rects(raster, position, size, self.PLAYER_ID)

    def _render_objects(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        raster = self._render_object_group(
            raster,
            state.diamond_x,
            state.diamond_y,
            state.diamond_active,
            self.consts.DIAMOND_WIDTH,
            self.consts.DIAMOND_HEIGHT,
            self.DIAMOND_ID,
        )
        raster = self._render_object_group(
            raster,
            state.enemy_x,
            state.enemy_y,
            state.enemy_active,
            self.consts.ENEMY_WIDTH,
            self.consts.ENEMY_HEIGHT,
            self.ENEMY_ID,
        )
        return self._render_object_group(
            raster,
            state.bullet_x,
            state.bullet_y,
            state.bullet_active,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            self.BULLET_ID,
        )

    def _render_object_group(
        self,
        raster: jnp.ndarray,
        x: chex.Array,
        y: chex.Array,
        active: chex.Array,
        width: int,
        height: int,
        color_id: int,
    ) -> jnp.ndarray:
        draw_x = jnp.where(active, jnp.round(x).astype(jnp.int32), -1)
        draw_y = jnp.round(y).astype(jnp.int32)
        positions = jnp.stack([draw_x, draw_y], axis=1)
        sizes = jnp.stack(
            [
                jnp.full(x.shape, width, dtype=jnp.int32),
                jnp.full(y.shape, height, dtype=jnp.int32),
            ],
            axis=1,
        )
        return self.jr.draw_rects(raster, positions, sizes, color_id)
