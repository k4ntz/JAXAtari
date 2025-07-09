import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

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
    all_rewards: chex.Array


class JaxPong(JaxEnvironment[PongState, PongObservation, PongInfo, PongConstants]):
    def __init__(self, consts: PongConstants = None, reward_funcs: list[callable]=None):
        consts = consts or PongConstants()
        super().__init__(consts)
        self.renderer = PongRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ]
        self.obs_size = 3*4+1+1

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state_player_y, state_player_speed, acceleration_counter, action: chex.Array):
        up = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        down = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

        acceleration = self.consts.PLAYER_ACCELERATION[acceleration_counter]

        touches_wall = jnp.logical_or(
            state_player_y < self.consts.WALL_TOP_Y,
            state_player_y + self.consts.PLAYER_SIZE[1] > self.consts.WALL_BOTTOM_Y,
        )

        player_speed = state_player_speed

        player_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_not(jnp.logical_or(up, down)), touches_wall),
            lambda s: jnp.round(s / 2).astype(jnp.int32),
            lambda s: s,
            operand=player_speed,
        )

        direction_change_up = jnp.logical_and(up, state_player_speed > 0)
        player_speed = jax.lax.cond(
            direction_change_up,
            lambda s: 0,
            lambda s: s,
            operand=player_speed,
        )
        direction_change_down = jnp.logical_and(down, state_player_speed < 0)

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
            operand=acceleration_counter,
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

        player_y = jnp.clip(
            state_player_y + player_speed,
            self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT - 10,
            self.consts.WALL_BOTTOM_Y - 4,
        )
        return player_y, player_speed, new_acceleration_counter

    def _ball_step(self, state: PongState, action):
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

        return ball_x, ball_y, ball_vel_x, ball_vel_y

    def _enemy_step(self, state, step_counter, ball_y, ball_speed_y):
        should_move = step_counter % 8 != 0

        direction = jnp.sign(ball_y - state.enemy_y)

        new_y = state.enemy_y + (direction * self.consts.ENEMY_STEP_SIZE).astype(jnp.int32)
        return jax.lax.cond(
            should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
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

    def reset(self, key=None) -> Tuple[PongObservation, PongState]:
        state = PongState(
            player_y=jnp.array(96).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            ball_x=jnp.array(78).astype(jnp.int32),
            ball_y=jnp.array(115).astype(jnp.int32),
            enemy_y=jnp.array(115).astype(jnp.int32),
            enemy_speed=jnp.array(0.0).astype(jnp.int32),
            ball_vel_x=self.consts.BALL_SPEED[0].astype(jnp.int32),
            ball_vel_y=self.consts.BALL_SPEED[1].astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            buffer=jnp.array(96).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PongState, action: chex.Array) -> Tuple[PongObservation, PongState, float, bool, PongInfo]:
        new_player_y, player_speed_b, new_acceleration_counter = self._player_step(
            state.player_y, state.player_speed, state.acceleration_counter, action
        )

        new_player_y, player_speed, new_acceleration_counter = jax.lax.cond(
            state.step_counter % 2 == 0,
            lambda _: (new_player_y, player_speed_b, new_acceleration_counter),
            lambda _: (state.player_y, state.player_speed, state.acceleration_counter),
            operand=None,
        )

        buffer = jax.lax.cond(
            jax.lax.eq(state.buffer, state.player_y),
            lambda _: new_player_y,
            lambda _: state.buffer,
            operand=None,
        )
        player_y = state.buffer

        enemy_y = self._enemy_step(state, state.step_counter, state.ball_y, state.ball_y)

        ball_x, ball_y, ball_vel_x, ball_vel_y = self._ball_step(state, action)

        player_goal = ball_x < 4
        enemy_goal = ball_x > 156
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
            ball_x.astype(jnp.int32),
            ball_y.astype(jnp.int32),
            ball_vel_x.astype(jnp.int32),
            ball_vel_y.astype(jnp.int32),
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
            lambda s: enemy_y.astype(jnp.int32),
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

        new_state = PongState(
            player_y=player_y,
            player_speed=player_speed,
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            enemy_y=enemy_y_final,
            enemy_speed=0,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=step_counter,
            acceleration_counter=new_acceleration_counter,
            buffer=buffer,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info


    def render(self, state: PongState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
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
        return spaces.Discrete(6)

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
    def _get_info(self, state: PongState, all_rewards: chex.Array = None) -> PongInfo:
        return PongInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        return (state.player_score - state.enemy_score) - (
            previous_state.player_score - previous_state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: PongState, state: PongState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PongState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.player_score, 21),
            jnp.greater_equal(state.enemy_score, 21),
        )


class PongRenderer(JAXGameRenderer):
    def __init__(self, consts: PongConstants = None):
        super().__init__()
        self.consts = consts or PongConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            self.SPRITE_BALL,
            self.PLAYER_DIGIT_SPRITES,
            self.ENEMY_DIGIT_SPRITES,
        ) = self.load_sprites()

    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/player.npy"))
        enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/enemy.npy"))
        ball = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/ball.npy"))

        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/background.npy"))

        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
        SPRITE_ENEMY = jnp.expand_dims(enemy, axis=0)
        SPRITE_BALL = jnp.expand_dims(ball, axis=0)

        PLAYER_DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/pong/player_score_{}.npy"),
            num_chars=10,
        )
        ENEMY_DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/pong/enemy_score_{}.npy"),
            num_chars=10,
        )

        return (
            SPRITE_BG,
            SPRITE_PLAYER,
            SPRITE_ENEMY,
            SPRITE_BALL,
            PLAYER_DIGIT_SPRITES,
            ENEMY_DIGIT_SPRITES
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = jr.render_at(raster, self.consts.PLAYER_X, state.player_y, frame_player)

        frame_enemy = jr.get_sprite_frame(self.SPRITE_ENEMY, 0)
        raster = jr.render_at(raster, self.consts.ENEMY_X, state.enemy_y, frame_enemy)

        frame_ball = jr.get_sprite_frame(self.SPRITE_BALL, 0)
        raster = jr.render_at(raster, state.ball_x, state.ball_y, frame_ball)

        # Direct wall rendering with HWC indexing
        wall_color = jnp.array(self.consts.WALL_COLOR, dtype=jnp.uint8)
        top_wall_y_start = self.consts.WALL_TOP_Y
        top_wall_y_end = self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT
        raster = raster.at[top_wall_y_start:top_wall_y_end, :, :].set(wall_color)

        bottom_wall_y_start = self.consts.WALL_BOTTOM_Y
        bottom_wall_y_end = self.consts.WALL_BOTTOM_Y + self.consts.WALL_BOTTOM_HEIGHT
        raster = raster.at[bottom_wall_y_start:bottom_wall_y_end, :, :].set(wall_color)

        player_score_digits = jr.int_to_digits(state.player_score, max_digits=2)
        enemy_score_digits = jr.int_to_digits(state.enemy_score, max_digits=2)

        is_player_single_digit = state.player_score < 10
        player_start_index = jax.lax.select(is_player_single_digit, 1, 0)
        player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)
        player_render_x = jax.lax.select(is_player_single_digit,
                                         120 + 16 // 2,
                                         120)

        raster = jr.render_label_selective(raster, player_render_x, 3,
                                            player_score_digits, self.PLAYER_DIGIT_SPRITES,
                                            player_start_index, player_num_to_render,
                                            spacing=16)

        is_enemy_single_digit = state.enemy_score < 10
        enemy_start_index = jax.lax.select(is_enemy_single_digit, 1, 0)
        enemy_num_to_render = jax.lax.select(is_enemy_single_digit, 1, 2)
        enemy_render_x = jax.lax.select(is_enemy_single_digit,
                                        10 + 16 // 2,
                                        10)

        raster = jr.render_label_selective(raster, enemy_render_x, 3,
                                           enemy_score_digits, self.ENEMY_DIGIT_SPRITES,
                                           enemy_start_index, enemy_num_to_render,
                                           spacing=16)

        return raster
