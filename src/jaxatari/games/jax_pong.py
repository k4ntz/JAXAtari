from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _create_wall_sprite(consts: "PongConstants", height: int) -> jnp.ndarray:
    wall_color_rgba = (*consts.SCORE_COLOR, 255)
    wall_shape = (height, consts.WIDTH, 4)
    return jnp.tile(jnp.array(wall_color_rgba, dtype=jnp.uint8), (*wall_shape[:2], 1))

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Pong.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},
        {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
        {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
        {'name': 'player_digits', 'type': 'digits', 'pattern': 'player_score_{}.npy'},
        {'name': 'enemy_digits', 'type': 'digits', 'pattern': 'enemy_score_{}.npy'},
    )


class PongConstants(NamedTuple):
    MAX_SPEED: int = 12
    BALL_SPEED: chex.Array = jnp.array([-1, 1])
    ENEMY_STEP_SIZE: int = 2
    WIDTH: int = 160
    HEIGHT: int = 210
    BASE_BALL_SPEED: int = 1
    BALL_MAX_SPEED: int = 4
    MIN_BALL_SPEED: int = 1
    PLAYER_ACCELERATION: chex.Array = jnp.array([6, 3, 1, -1, 1, -1, 0, 0, 1, 0, -1, 0, 1])
    BALL_START_X: chex.Array = jnp.array(78)
    BALL_START_Y: chex.Array = jnp.array(115)
    BACKGROUND_COLOR: Tuple[int, int, int] = (144, 72, 17)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    ENEMY_COLOR: Tuple[int, int, int] = (213, 130, 74)
    BALL_COLOR: Tuple[int, int, int] = (236, 236, 236)
    WALL_COLOR: Tuple[int, int, int] = (236, 236, 236)
    SCORE_COLOR: Tuple[int, int, int] = (236, 236, 236)
    PLAYER_X: int = 140
    ENEMY_X: int = 16
    PLAYER_SIZE: Tuple[int, int] = (4, 16)
    BALL_SIZE: Tuple[int, int] = (2, 4)
    ENEMY_SIZE: Tuple[int, int] = (4, 16)
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16
    # sset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()


# immutable state container
class PongState(NamedTuple):
    player_y: chex.Array
    player_speed: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    enemy_y: chex.Array
    enemy_speed: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    acceleration_counter: chex.Array
    buffer: chex.Array
    key: chex.PRNGKey


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class PongObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    ball: EntityPosition
    score_player: jnp.ndarray
    score_enemy: jnp.ndarray


class PongInfo(NamedTuple):
    time: jnp.ndarray


class JaxPong(JaxEnvironment[PongState, PongObservation, PongInfo, PongConstants]):
    # Minimal ALE action set for Pong:
    # 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )

    def __init__(self, consts: PongConstants = None):
        consts = consts or PongConstants()
        super().__init__(consts)
        self.renderer = PongRenderer(self.consts)

    def _player_step(self, state: PongState, action: chex.Array) -> PongState:
        up = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        down = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

        acceleration = self.consts.PLAYER_ACCELERATION[state.acceleration_counter]

        touches_wall = jnp.logical_or(
            state.player_y < self.consts.WALL_TOP_Y,
            state.player_y + self.consts.PLAYER_SIZE[1] > self.consts.WALL_BOTTOM_Y,
        )

        player_speed = state.player_speed

        player_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_not(jnp.logical_or(up, down)), touches_wall),
            lambda s: jnp.round(s / 2).astype(jnp.int32),
            lambda s: s,
            operand=player_speed,
        )

        direction_change_up = jnp.logical_and(up, state.player_speed > 0)
        player_speed = jax.lax.cond(
            direction_change_up,
            lambda s: 0,
            lambda s: s,
            operand=player_speed,
        )
        direction_change_down = jnp.logical_and(down, state.player_speed < 0)

        player_speed = jax.lax.cond(
            direction_change_down,
            lambda s: 0,
            lambda s: s,
            operand=player_speed,
        )

        direction_change = jnp.logical_or(direction_change_up, direction_change_down)
        acceleration_counter = jax.lax.cond(
            direction_change,
            lambda _: 0,
            lambda s: s,
            operand=state.acceleration_counter,
        )

        player_speed = jax.lax.cond(
            up,
            lambda s: jnp.maximum(s - acceleration, -self.consts.MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        player_speed = jax.lax.cond(
            down,
            lambda s: jnp.minimum(s + acceleration, self.consts.MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        new_acceleration_counter = jax.lax.cond(
            jnp.logical_or(up, down),
            lambda s: jnp.minimum(s + 1, 15),
            lambda s: 0,
            operand=acceleration_counter,
        )

        proposed_player_y = jnp.clip(
            state.player_y + player_speed,
            self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT - 10,
            self.consts.WALL_BOTTOM_Y - 4,
        )

        # Match original timing/buffering behavior
        new_player_y, new_player_speed, new_acc_counter = jax.lax.cond(
            state.step_counter % 2 == 0,
            lambda _: (proposed_player_y, player_speed, new_acceleration_counter),
            lambda _: (state.player_y, state.player_speed, state.acceleration_counter),
            operand=None,
        )

        buffer = jax.lax.cond(
            jax.lax.eq(state.buffer, state.player_y),
            lambda _: new_player_y,
            lambda _: state.buffer,
            operand=None,
        )
        final_player_y = state.buffer

        return PongState(
            player_y=final_player_y,
            player_speed=new_player_speed,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            enemy_y=state.enemy_y,
            enemy_speed=state.enemy_speed,
            ball_vel_x=state.ball_vel_x,
            ball_vel_y=state.ball_vel_y,
            player_score=state.player_score,
            enemy_score=state.enemy_score,
            step_counter=state.step_counter,
            acceleration_counter=new_acc_counter,
            buffer=buffer,
            key=state.key,
        )

    def _ball_step(self, state: PongState, action) -> PongState:
        ball_x = state.ball_x + state.ball_vel_x
        ball_y = state.ball_y + state.ball_vel_y

        wall_bounce = jnp.logical_or(
            ball_y <= self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT - self.consts.BALL_SIZE[1],
            ball_y >= self.consts.WALL_BOTTOM_Y,
        )
        ball_vel_y = jnp.where(wall_bounce, -state.ball_vel_y, state.ball_vel_y)

        player_paddle_hit = jnp.logical_and(
            jnp.logical_and(self.consts.PLAYER_X <= ball_x, ball_x <= self.consts.PLAYER_X + self.consts.PLAYER_SIZE[0]),
            state.ball_vel_x > 0,
        )

        player_paddle_hit = jnp.logical_and(
            player_paddle_hit,
            jnp.logical_and(
                state.player_y - self.consts.BALL_SIZE[1] <= ball_y,
                ball_y <= state.player_y + self.consts.PLAYER_SIZE[1] + self.consts.BALL_SIZE[1],
            ),
        )

        enemy_paddle_hit = jnp.logical_and(
            jnp.logical_and(self.consts.ENEMY_X <= ball_x, ball_x <= self.consts.ENEMY_X + self.consts.ENEMY_SIZE[0] - 1),
            state.ball_vel_x < 0,
        )

        enemy_paddle_hit = jnp.logical_and(
            enemy_paddle_hit,
            jnp.logical_and(
                state.enemy_y - self.consts.BALL_SIZE[1] <= ball_y,
                ball_y <= state.enemy_y + self.consts.ENEMY_SIZE[1] + self.consts.BALL_SIZE[1],
            ),
        )

        paddle_hit = jnp.logical_or(player_paddle_hit, enemy_paddle_hit)

        section_height = self.consts.PLAYER_SIZE[1] / 5

        hit_position = jnp.where(
            paddle_hit,
            jnp.where(
                player_paddle_hit,
                jnp.where(
                    ball_y < state.player_y + section_height,
                    -2.0,
                    jnp.where(
                        ball_y < state.player_y + 2 * section_height,
                        -1.0,
                        jnp.where(
                            ball_y < state.player_y + 3 * section_height,
                            0.0,
                            jnp.where(
                                ball_y < state.player_y + 4 * section_height,
                                1.0,
                                2.0,
                            ),
                        ),
                    ),
                ),
                jnp.where(
                    ball_y < state.enemy_y + section_height,
                    -2.0,
                    jnp.where(
                        ball_y < state.enemy_y + 2 * section_height,
                        -1.0,
                        jnp.where(
                            ball_y < state.enemy_y + 3 * section_height,
                            0.0,
                            jnp.where(
                                ball_y < state.enemy_y + 4 * section_height,
                                1.0,
                                2.0,
                            ),
                        ),
                    ),
                ),
            ),
            0.0,
        )

        paddle_speed = jnp.where(
            player_paddle_hit,
            state.player_speed,
            jnp.where(
                enemy_paddle_hit,
                state.enemy_speed,
                0.0,
            ),
        )

        ball_vel_y = jnp.where(paddle_hit, hit_position, ball_vel_y)

        boost_triggered = jnp.logical_and(
            player_paddle_hit,
            jnp.logical_or(
                jnp.logical_or(action == Action.LEFTFIRE, action == Action.RIGHTFIRE),
                action == Action.FIRE,
            ),
        )
        player_max_hit = jnp.logical_and(player_paddle_hit, state.player_speed == self.consts.MAX_SPEED)
        ball_vel_x = jnp.where(
            jnp.logical_or(boost_triggered, player_max_hit),
            state.ball_vel_x
            + jnp.sign(state.ball_vel_x),
            state.ball_vel_x,
        )

        ball_vel_x = jnp.where(
            paddle_hit,
            -ball_vel_x,
            ball_vel_x,
        )

        return PongState(
            player_y=state.player_y,
            player_speed=state.player_speed,
            ball_x=ball_x.astype(jnp.int32),
            ball_y=ball_y.astype(jnp.int32),
            enemy_y=state.enemy_y,
            enemy_speed=state.enemy_speed,
            ball_vel_x=ball_vel_x.astype(jnp.int32),
            ball_vel_y=ball_vel_y.astype(jnp.int32),
            player_score=state.player_score,
            enemy_score=state.enemy_score,
            step_counter=state.step_counter,
            acceleration_counter=state.acceleration_counter,
            buffer=state.buffer,
            key=state.key,
        )

    def _enemy_step(self, state: PongState) -> PongState:
        should_move = state.step_counter % 8 != 0

        direction = jnp.sign(state.ball_y - state.enemy_y)

        new_y = state.enemy_y + (direction * self.consts.ENEMY_STEP_SIZE).astype(jnp.int32)
        enemy_y = jax.lax.cond(
            should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None
        )
        return PongState(
            player_y=state.player_y,
            player_speed=state.player_speed,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            enemy_y=enemy_y.astype(jnp.int32),
            enemy_speed=state.enemy_speed,
            ball_vel_x=state.ball_vel_x,
            ball_vel_y=state.ball_vel_y,
            player_score=state.player_score,
            enemy_score=state.enemy_score,
            step_counter=state.step_counter,
            acceleration_counter=state.acceleration_counter,
            buffer=state.buffer,
            key=state.key,
        )

    def _score_and_reset(self, state: PongState) -> PongState:
        player_goal = state.ball_x < 4
        enemy_goal = state.ball_x > 156
        ball_reset = jnp.logical_or(enemy_goal, player_goal)

        player_score = jax.lax.cond(
            player_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.player_score,
        )
        enemy_score = jax.lax.cond(
            enemy_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.enemy_score,
        )

        current_values = (
            state.ball_x.astype(jnp.int32),
            state.ball_y.astype(jnp.int32),
            state.ball_vel_x.astype(jnp.int32),
            state.ball_vel_y.astype(jnp.int32),
        )
        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: self._reset_ball_after_goal((state, enemy_goal)),
            lambda x: x,
            operand=current_values,
        )

        step_counter = jax.lax.cond(
            ball_reset,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        enemy_y_final = jax.lax.cond(
            ball_reset,
            lambda s: self.consts.BALL_START_Y.astype(jnp.int32),
            lambda s: state.enemy_y.astype(jnp.int32),
            operand=None,
        )

        ball_x_final = jax.lax.cond(
            step_counter < 60,
            lambda s: self.consts.BALL_START_X.astype(jnp.int32),
            lambda s: s,
            operand=ball_x_final,
        )
        ball_y_final = jax.lax.cond(
            step_counter < 60,
            lambda s: self.consts.BALL_START_Y.astype(jnp.int32),
            lambda s: s,
            operand=ball_y_final,
        )

        return PongState(
            player_y=state.player_y,
            player_speed=state.player_speed,
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            enemy_y=enemy_y_final,
            enemy_speed=state.enemy_speed,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=step_counter,
            acceleration_counter=state.acceleration_counter,
            buffer=state.buffer,
            key=state.key,
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PongState, action: chex.Array) -> Tuple[PongObservation, PongState, float, bool, PongInfo]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        previous_state = state
        state = self._player_step(state, atari_action)
        state = self._enemy_step(state)
        state = self._ball_step(state, atari_action)
        state = self._score_and_reset(state)

    def _reset_ball_after_goal(self, state_and_goal: Tuple[PongState, bool]) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        state, scored_right = state_and_goal

        ball_vel_y = jnp.where(
            state.ball_y > self.consts.BALL_START_Y,
            1,
            -1,
        ).astype(jnp.int32)

        ball_vel_x = jnp.where(
            scored_right, 1, -1
        ).astype(jnp.int32)

        return (
            self.consts.BALL_START_X.astype(jnp.int32),
            self.consts.BALL_START_Y.astype(jnp.int32),
            ball_vel_x.astype(jnp.int32),
            ball_vel_y.astype(jnp.int32),
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PongObservation, PongState]:
        # Split key for env reset if needed and for state storage
        state_key, _step_key = jax.random.split(key)
        state = PongState(
            player_y=jnp.array(96).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            ball_x=self.consts.BALL_START_X.astype(jnp.int32),
            ball_y=self.consts.BALL_START_Y.astype(jnp.int32),
            enemy_y=jnp.array(115).astype(jnp.int32),
            enemy_speed=jnp.array(0.0).astype(jnp.int32),
            ball_vel_x=self.consts.BALL_SPEED[0].astype(jnp.int32),
            ball_vel_y=self.consts.BALL_SPEED[1].astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            buffer=jnp.array(96).astype(jnp.int32),
            key=state_key,
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PongState, action: chex.Array) -> Tuple[PongObservation, PongState, float, bool, PongInfo]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Split step key from state and keep a new key for the next state
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state
        # Make per-step key available to helpers that may read state.key
        state = PongState(
            player_y=state.player_y,
            player_speed=state.player_speed,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            enemy_y=state.enemy_y,
            enemy_speed=state.enemy_speed,
            ball_vel_x=state.ball_vel_x,
            ball_vel_y=state.ball_vel_y,
            player_score=state.player_score,
            enemy_score=state.enemy_score,
            step_counter=state.step_counter,
            acceleration_counter=state.acceleration_counter,
            buffer=state.buffer,
            key=step_key,
        )
        state = self._player_step(state, atari_action)
        state = self._enemy_step(state)
        state = self._ball_step(state, atari_action)
        state = self._score_and_reset(state)

        # Update state key to new_state_key for next step
        state = state._replace(key=new_state_key)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: PongState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: PongState):
        player = EntityPosition(
            x=jnp.array(self.consts.PLAYER_X),
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        enemy = EntityPosition(
            x=jnp.array(self.consts.ENEMY_X),
            y=state.enemy_y,
            width=jnp.array(self.consts.ENEMY_SIZE[0]),
            height=jnp.array(self.consts.ENEMY_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(self.consts.BALL_SIZE[0]),
            height=jnp.array(self.consts.BALL_SIZE[1]),
        )
        return PongObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            score_player=state.player_score,
            score_enemy=state.enemy_score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: PongObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten(),
               obs.enemy.x.flatten(),
               obs.enemy.y.flatten(),
               obs.enemy.height.flatten(),
               obs.enemy.width.flatten(),
               obs.ball.x.flatten(),
               obs.ball.y.flatten(),
               obs.ball.height.flatten(),
               obs.ball.width.flatten(),
               obs.score_player.flatten(),
               obs.score_enemy.flatten()
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "enemy": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "ball": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "score_player": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
            "score_enemy": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PongState, ) -> PongInfo:
        return PongInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        return (state.player_score - state.enemy_score) - (
            previous_state.player_score - previous_state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PongState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.player_score, 21),
            jnp.greater_equal(state.enemy_score, 21),
        )

class PongRenderer(JAXGameRenderer):
    def __init__(self, consts: PongConstants = None):
        super().__init__(consts)
        self.consts = consts or PongConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 2. Create procedural assets using modded constants
        wall_sprite_top = _create_wall_sprite(self.consts, self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = _create_wall_sprite(self.consts, self.consts.WALL_BOTTOM_HEIGHT)

        # 3. Append procedural assets
        final_asset_config.append({'name': 'wall_top', 'type': 'procedural', 'data': wall_sprite_top})
        final_asset_config.append({'name': 'wall_bottom', 'type': 'procedural', 'data': wall_sprite_bottom})

        # 4. Bake assets once
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/pong"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, self.consts.PLAYER_X, state.player_y, player_mask)

        enemy_mask = self.SHAPE_MASKS["enemy"]
        raster = self.jr.render_at(raster, self.consts.ENEMY_X, state.enemy_y, enemy_mask)

        ball_mask = self.SHAPE_MASKS["ball"]
        raster = self.jr.render_at(raster, state.ball_x, state.ball_y, ball_mask)

        # --- Stamp Walls and Score (using the same color/ID) ---
        score_color_tuple = self.consts.SCORE_COLOR # (236, 236, 236)
        score_id = self.COLOR_TO_ID[score_color_tuple]

        # Draw walls (using separate sprites for top and bottom)
        raster = self.jr.render_at(raster, 0, self.consts.WALL_TOP_Y, self.SHAPE_MASKS["wall_top"])
        raster = self.jr.render_at(raster, 0, self.consts.WALL_BOTTOM_Y, self.SHAPE_MASKS["wall_bottom"])

        # Stamp Score using the label utility
        player_digits = self.jr.int_to_digits(state.player_score, max_digits=2)
        enemy_digits = self.jr.int_to_digits(state.enemy_score, max_digits=2)

        # Note: The logic for single/double digits is complex for a jitted function.
        player_digit_masks = self.SHAPE_MASKS["player_digits"] # Assumes single color
        enemy_digit_masks = self.SHAPE_MASKS["enemy_digits"] # Assumes single color

        is_player_single_digit = state.player_score < 10
        player_start_index = jax.lax.select(is_player_single_digit, 1, 0)
        player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)
        player_render_x = jax.lax.select(is_player_single_digit,
                                         120 + 16 // 2,
                                         120)

        raster = self.jr.render_label_selective(raster, player_render_x, 3, player_digits, player_digit_masks, player_start_index, player_num_to_render, spacing=16)
        
        is_enemy_single_digit = state.enemy_score < 10
        enemy_start_index = jax.lax.select(is_enemy_single_digit, 1, 0)
        enemy_num_to_render = jax.lax.select(is_enemy_single_digit, 1, 2)
        enemy_render_x = jax.lax.select(is_enemy_single_digit,
                                        10 + 16 // 2,
                                        10)

        raster = self.jr.render_label_selective(raster, enemy_render_x, 3, enemy_digits, enemy_digit_masks, enemy_start_index, enemy_num_to_render, spacing=16)

        return self.jr.render_from_palette(raster, self.PALETTE)