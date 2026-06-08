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
    pass


@struct.dataclass
class JamesBondObservation:
    pass


@struct.dataclass
class JamesBondInfo:
    pass


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
        raise NotImplementedError("JamesBond reset is added in a later commit")

    def step(
        self, state: JamesBondState, action: chex.Array
    ) -> Tuple[JamesBondObservation, JamesBondState, chex.Array, chex.Array, JamesBondInfo]:
        raise NotImplementedError("JamesBond step is added in a later commit")

    def render(self, state: JamesBondState) -> jnp.ndarray:
        raise NotImplementedError("JamesBond renderer is added in a later commit")

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({})

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)
