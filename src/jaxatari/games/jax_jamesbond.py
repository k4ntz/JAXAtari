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


    def render(self, state: JamesBondState) -> jnp.ndarray:
        raise NotImplementedError("JamesBond renderer is added in a later commit")

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({})

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

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

