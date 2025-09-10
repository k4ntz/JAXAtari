from dataclasses import dataclass
from jaxatari.renderers import JAXGameRenderer
from typing import NamedTuple, Tuple
from jax import Array, jit, random, numpy as jnp, debug
from functools import partial
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
import jax.lax


class TronConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3
    player_height: int = 10
    player_width: int = 10
    player_speed: int = 1


class Actors(NamedTuple):
    x: Array  # (N,) int32: X-Position
    y: Array  # (N,) int32: Y-Position
    vx: Array  # (N,) int32: velocity in x-direction
    vy: Array  # (N,) int32: velocity in y-direction
    w: Array  # (N,) int32: Width
    h: Array  # (N,) int32: Height
    alive: Array  # (N,) int32: Boolean mask


class TronState(NamedTuple):
    score: Array
    player: Actors  # N = 1
    # enemies: Actors # N = MAX_ENEMIES


class EntityPosition(NamedTuple):
    x: Array
    y: Array
    width: Array
    height: Array


class TronObservation(NamedTuple):
    pass


class TronInfo(NamedTuple):
    pass


class TronRenderer(JAXGameRenderer):
    def __init__(self, consts: TronConstants = None) -> None:
        super().__init__()
        self.consts = consts or TronConstants()

    @partial(jit, static_argnums=(0,))
    def render(self, state) -> Array:
        raster = jr.create_initial_frame(
            width=self.consts.screen_width, height=self.consts.screen_height
        )
        blue_color = jnp.array([0, 0, 255, 255], dtype=jnp.uint8)
        blue_box_sprite = jnp.broadcast_to(blue_color, (10, 10, 4))
        player_x, player_y = state.player.x[0], state.player.y[0]
        raster = jr.render_at(
            raster,
            player_x,
            player_y,
            blue_box_sprite,
        )
        return raster


class JaxTron(
    JaxEnvironment[TronState, TronObservation, TronInfo, TronConstants]
):
    def __init__(
        self, consts: TronConstants = None, reward_funcs: list[callable] = None
    ) -> None:
        consts = consts or TronConstants()
        super().__init__(consts)
        self.renderer = TronRenderer
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
            Action.DOWNLEFTFIRE,
        ]

    def reset(
        self, key: random.PRNGKey = None
    ) -> Tuple[TronObservation, TronState]:
        def _get_centered_player_pos(consts: TronConstants) -> Actors:
            W, H = consts.screen_width, consts.screen_height
            w, h = consts.player_width, consts.player_height
            x0 = (W - w) // 2
            y0 = (H - h) // 2
            return Actors(
                x=jnp.array([x0], dtype=jnp.int32),
                y=jnp.array([y0], dtype=jnp.int32),
                vx=jnp.zeros(1, dtype=jnp.int32),
                vy=jnp.zeros(1, dtype=jnp.int32),
                w=jnp.full((1,), w, dtype=jnp.int32),
                h=jnp.full((1,), h, dtype=jnp.int32),
                alive=jnp.array([True], dtype=jnp.bool_),
            )

        debug.print("{X}", X=_get_centered_player_pos(self.consts).vx)

        new_state: TronState = TronState(
            score=jnp.zeros((), dtype=jnp.int32),
            player=_get_centered_player_pos(self.consts),
        )
        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jit, static_argnums=(0,))
    def _player_step(self, state: TronState, action: Array) -> TronState:
        player = state.player
        speed = self.consts.player_speed

        # boolean flags for primary direction
        is_moving_up = (
            (action == Action.UP)
            | (action == Action.UPRIGHT)
            | (action == Action.UPLEFT)
            | (action == Action.UPFIRE)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.UPLEFTFIRE)
        )

        is_moving_down = (
            (action == Action.DOWN)
            | (action == Action.DOWNRIGHT)
            | (action == Action.DOWNLEFT)
            | (action == Action.DOWNFIRE)
            | (action == Action.DOWNRIGHTFIRE)
            | (action == Action.DOWNLEFTFIRE)
        )

        is_moving_right = (
            (action == Action.RIGHT)
            | (action == Action.UPRIGHT)
            | (action == Action.DOWNRIGHT)
            | (action == Action.RIGHTFIRE)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.DOWNRIGHTFIRE)
        )

        is_moving_left = (
            (action == Action.LEFT)
            | (action == Action.UPLEFT)
            | (action == Action.DOWNLEFT)
            | (action == Action.LEFTFIRE)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.DOWNLEFTFIRE)
        )

        # determine the velocity with the previous binary values.
        # combined they enable diagonal movement
        dx_scalar = jnp.where(
            is_moving_right, speed, jnp.where(is_moving_left, -speed, 0)
        )

        dy_scalar = jnp.where(
            is_moving_up, -speed, jnp.where(is_moving_down, speed, 0)
        )

        # calculate new positions
        nx = player.x[0] + dx_scalar
        ny = player.y[0] + dy_scalar

        # clamp to screen boundaries
        # TODO: Change later to inner boundary
        max_x = self.consts.screen_width - player.w[0]
        max_y = self.consts.screen_height - player.h[0]
        nx = jnp.clip(nx, 0, max_x)
        ny = jnp.clip(ny, 0, max_y)

        new_player = player._replace(
            x=jnp.array([nx], dtype=jnp.int32),
            y=jnp.array([ny], dtype=jnp.int32),
            vx=jnp.array([dx_scalar], dtype=jnp.int32),
            vy=jnp.array([dy_scalar], dtype=jnp.int32),
        )
        return state._replace(player=new_player)

    @partial(jit, static_argnums=(0,))
    def step(
        self, state: TronState, action: Array
    ) -> Tuple[TronObservation, TronState, float, bool, TronInfo]:

        new_state: TronState = self._player_step(state, action)

        obs: TronObservation = self._get_observation(new_state)
        env_reward: float = self._get_reward(state, new_state)
        done: bool = self._get_done(new_state)
        info: TronInfo = self._get_info(state)

        return obs, new_state, env_reward, done, info

    @partial(jit, static_argnums=(0,))
    def _get_observation(self, state: TronState) -> TronObservation:
        return TronObservation()

    @partial(jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TronState, state: TronState) -> float:
        return 0.0

    @partial(jit, static_argnums=(0,))
    def _get_done(self, state: TronState) -> bool:
        return False

    @partial(jit, static_argnums=(0,))
    def _get_info(self, state: TronState) -> TronInfo:
        return TronInfo()

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
