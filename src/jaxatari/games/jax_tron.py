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

    # discs
    max_discs: int = 4
    disc_size: int = 4
    disc_speed: int = 2


class Actors(NamedTuple):
    x: Array  # (N,) int32: X-Position
    y: Array  # (N,) int32: Y-Position
    vx: Array  # (N,) int32: velocity in x-direction
    vy: Array  # (N,) int32: velocity in y-direction
    w: Array  # (N,) int32: Width
    h: Array  # (N,) int32: Height


class Enemies(Actors):
    alive: Array  # (N,) int32: Boolean mask


class Discs(Actors):
    owner: Array  # (D,) int32, 0 = player, 1 = enemy
    phase: Array  # (D,) int32, 0=idle/unused, 1=outbound, 2=returning (player only)


class TronState(NamedTuple):
    score: Array
    player: Actors  # N = 1
    # enemies: Actors # N = MAX_ENEMIES
    # short cooldown after each wave/color
    wave_end_cooldown_remaining: Array
    aim_dx: Array  # remember last movement direction in X-dir
    aim_dy: Array  # remember last movement direction in Y-dir
    # discs: Discs


class EntityPosition(NamedTuple):
    x: Array
    y: Array
    width: Array
    height: Array


class TronObservation(NamedTuple):
    pass


class TronInfo(NamedTuple):
    pass


class UserAction(NamedTuple):
    """Boolean flags for the players action"""

    up: Array
    down: Array
    left: Array
    right: Array
    fire: Array
    moved: Array  # flag for any movement


@jit
def parse_action(action: Array) -> UserAction:
    """Translate the raw action integer into a UserAction"""
    is_up = (
        (action == Action.UP)
        | (action == Action.UPRIGHT)
        | (action == Action.UPLEFT)
        | (action == Action.UPFIRE)
        | (action == Action.UPRIGHTFIRE)
        | (action == Action.UPLEFTFIRE)
    )

    is_down = (
        (action == Action.DOWN)
        | (action == Action.DOWNRIGHT)
        | (action == Action.DOWNLEFT)
        | (action == Action.DOWNFIRE)
        | (action == Action.DOWNRIGHTFIRE)
        | (action == Action.DOWNLEFTFIRE)
    )

    is_right = (
        (action == Action.RIGHT)
        | (action == Action.UPRIGHT)
        | (action == Action.DOWNRIGHT)
        | (action == Action.RIGHTFIRE)
        | (action == Action.UPRIGHTFIRE)
        | (action == Action.DOWNRIGHTFIRE)
    )

    is_left = (
        (action == Action.LEFT)
        | (action == Action.UPLEFT)
        | (action == Action.DOWNLEFT)
        | (action == Action.LEFTFIRE)
        | (action == Action.UPLEFTFIRE)
        | (action == Action.DOWNLEFTFIRE)
    )

    is_fire = (
        (action == Action.FIRE)
        | (action == Action.UPFIRE)
        | (action == Action.RIGHTFIRE)
        | (action == Action.LEFTFIRE)
        | (action == Action.DOWNFIRE)
        | (action == Action.UPRIGHTFIRE)
        | (action == Action.UPLEFTFIRE)
        | (action == Action.DOWNRIGHTFIRE)
    )

    # The moved flag is just an OR of the directions
    has_moved = is_up | is_down | is_left | is_right

    return UserAction(
        up=is_up,
        down=is_down,
        left=is_left,
        right=is_right,
        fire=is_fire,
        moved=has_moved,
    )


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


####
# Helper functions
####
@jit
def set_velocity(actors: Actors, vx: Array, vy: Array) -> Actors:
    """Returns a new Actors instanc ce with updated velocity"""
    return actors._replace(vx=vx, vy=vy)


@jit
def move(actors: Actors) -> Actors:
    """Returns a new Actors instance with positions updated by velocity"""
    return actors._replace(x=actors.x + actors.vx, y=actors.y + actors.vy)


@jit
def clamp_position(actors: Actors, max_x: int, max_y: int) -> Actors:
    """Clamps positions according to the given max_x and max_y"""
    # TODO: Change later to inner boundary
    new_x = jnp.clip(actors.x, 0, max_x - actors.w)
    new_y = jnp.clip(actors.y, 0, max_y - actors.h)
    return actors._replace(x=new_x, y=new_y)


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
            screen_w, screen_h = consts.screen_width, consts.screen_height
            player_w, player_h = consts.player_width, consts.player_height
            x0 = (screen_w - player_w) // 2
            y0 = (screen_h - player_h) // 2
            return Actors(
                x=jnp.array([x0], dtype=jnp.int32),
                y=jnp.array([y0], dtype=jnp.int32),
                vx=jnp.zeros(1, dtype=jnp.int32),
                vy=jnp.zeros(1, dtype=jnp.int32),
                w=jnp.full((1,), player_w, dtype=jnp.int32),
                h=jnp.full((1,), player_h, dtype=jnp.int32),
            )

        debug.print("{X}", X=_get_centered_player_pos(self.consts).y)

        new_state: TronState = TronState(
            score=jnp.zeros((), dtype=jnp.int32),
            player=_get_centered_player_pos(self.consts),
            wave_end_cooldown_remaining=jnp.zeros((), dtype=jnp.int32),
            aim_dx=jnp.zeros((), dtype=jnp.int32),
            aim_dy=jnp.zeros((), dtype=jnp.int32),
        )
        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def _cooldown_finished(self, state: TronState) -> Array:
        # TODO: Change later to ==
        return state.wave_end_cooldown_remaining != 0

    @partial(jit, static_argnums=(0,))
    def _player_step(self, state: TronState, action: Array) -> TronState:
        player = state.player
        speed = self.consts.player_speed

        # parse the raw action to a structured object with boolean flags
        # e.g. down=True, right=True
        parsed_action = parse_action(action)

        # Calculate horizontal velocity
        # the boolean subtraction (right - left) results in +1, -1 or 0
        dx = speed * (
            parsed_action.right.astype(jnp.int32)
            - parsed_action.left.astype(jnp.int32)
        )

        # Calculate vertical velocity
        # the boolean subtraction (down - up) results in +1, -1 or 0
        dy = speed * (
            parsed_action.down.astype(jnp.int32)
            - parsed_action.up.astype(jnp.int32)
        )

        # Set the new velocity on the player actor
        player = set_velocity(player, dx, dy)
        # apply the velocity to the players position
        player = move(player)
        # Ensure the new position is without screen boundaries
        player = clamp_position(
            player, self.consts.screen_width, self.consts.screen_height
        )

        # only update the aiming direction, if movement key was pressed
        aim_dx = jnp.where(parsed_action.moved, dx, state.aim_dx)
        aim_dy = jnp.where(parsed_action.moved, dy, state.aim_dy)

        return state._replace(player=player, aim_dx=aim_dx, aim_dy=aim_dy)

    @partial(jit, static_argnums=(0,))
    def _check_action_fire(self, action: Array) -> Array:
        return (
            (action == Action.FIRE)
            | (action == Action.UPFIRE)
            | (action == Action.RIGHTFIRE)
            | (action == Action.LEFTFIRE)
            | (action == Action.DOWNFIRE)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.DOWNRIGHTFIRE)
            | (action == Action.DOWNLEFTFIRE)
        )

    @partial(jit, static_argnums=(0,))
    def _spawn_disc(self, state: TronState, action: Array) -> TronState:
        # check if the user pressed fire
        pressed_fire = self._check_action_fire(action)

        return state
        # check if the user

    @partial(jit, static_argnums=(0,))
    def step(
        self, state: TronState, action: Array
    ) -> Tuple[TronObservation, TronState, float, bool, TronInfo]:
        previous_state = state

        def _pause_step(s: TronState) -> TronState:
            # TODO: Activate later
            # s: TronState = s._replace(
            #    wave_end_cooldown_remaining=jnp.maximum(
            #        s.wave_end_cooldown_remaining - 1, 0
            #    )
            # )
            # s = self.move_discs(s)
            s: TronState = self._player_step(s, action)
            s: TronState = self._spawn_disc(s, action)
            return s

        def _wave_step(s: TronState) -> TronState:
            s: TronState = self._player_step(s, action)
            s: TronState = self._spawn_disc(s, action)
            return s

        state = jax.lax.cond(
            self._cooldown_finished(state), _wave_step, _pause_step, state
        )

        obs: TronObservation = self._get_observation(state)
        env_reward: float = self._get_reward(state, state)
        done: bool = self._get_done(state)
        info: TronInfo = self._get_info(state)

        return obs, state, env_reward, done, info

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
