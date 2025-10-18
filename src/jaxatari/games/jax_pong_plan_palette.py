import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import numpy as np

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import planned_render_save as jr
from jaxatari.rendering import planned_render_save_downscaled as jr_downscaled
from jaxatari.rendering import jax_rendering_utils_pallette as other_jr
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
        #return self.renderer_downscaled.render(state)

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


SPRITE_ID_PLAYER = 0
SPRITE_ID_ENEMY = 1
SPRITE_ID_BALL = 2
SPRITE_ID_WALL = 3 # -> A new sprite for the walls
# -> Digit sprites will start after the main ones.
SPRITE_ID_PLAYER_DIGITS_START = 4
SPRITE_ID_ENEMY_DIGITS_START = 14 # (4 + 10 player digits)


class PongRenderer(JAXGameRenderer):
    def __init__(self, consts: PongConstants = None):
        self.consts = consts or PongConstants()
        
        # Initialize assets for the planned palette renderer
        (
            self.palette,
            self.mask_atlas,
            self.sprite_info_table,
            self.background_raster
        ) = self._initialize_assets()
        
        # Store max sprite dimensions for the execution function (as static args)
        self.max_sprite_w = jnp.max(self.sprite_info_table[:, 2]).item()
        self.max_sprite_h = jnp.max(self.sprite_info_table[:, 3]).item()

    def _initialize_assets(self):
        """
        Loads all sprites, creates a palette, a mask atlas, and sprite info table.
        """
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Use a temporary instance of the palette utility to help with setup
        temp_jr = other_jr.JaxRenderingUtils(other_jr.RendererConfig())

        # 1. Load all raw RGBA sprites
        sprites_rgba = {
            "player": temp_jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/player.npy")),
            "enemy": temp_jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/enemy.npy")),
            "ball": temp_jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/ball.npy")),
            "background": temp_jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pong/background.npy")),
            "player_digits": temp_jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/pong/player_score_{}.npy")),
            "enemy_digits": temp_jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/pong/enemy_score_{}.npy")),
        }

        # 2. Discover palette and create color_to_id map from all sprites
        # This logic is borrowed from the palette utility's setup function
        color_to_id = {}
        palette_list = []
        next_id = 0

        for sprite_data in sprites_rgba.values():
            sprites_to_process = sprite_data if sprite_data.ndim >= 4 else jnp.expand_dims(sprite_data, axis=0)
            for sprite in sprites_to_process:
                pixels = np.array(sprite.reshape(-1, 4))
                for r, g, b, a in pixels:
                    if a > 0:
                        # Round to nearest integer to handle interpolation artifacts
                        rgb = (int(np.round(r)), int(np.round(g)), int(np.round(b)))
                        if rgb not in color_to_id:
                            color_to_id[rgb] = next_id
                            palette_list.append(rgb)
                            next_id += 1

        palette = jnp.array(palette_list, dtype=jnp.uint8)

        # 3. Convert RGBA sprites to Integer ID Masks
        id_masks = {
            name: jr._create_id_masks_from_batch(data, color_to_id) if data.ndim == 4
                else [temp_jr._create_id_mask(data, color_to_id)]
            for name, data in sprites_rgba.items() if name != "background"
        }

        # 4. Pack all ID masks into a single "Mask Atlas" and create the info table
        all_masks_ordered = (
            id_masks["player"] + id_masks["enemy"] + id_masks["ball"] +
            id_masks["player_digits"] + id_masks["enemy_digits"]
        )
        
        sprite_info = []
        current_v = 0
        for mask in all_masks_ordered:
            h, w = mask.shape
            sprite_info.append([0, current_v, w, h]) # u, v, width, height
            current_v += h

        max_width = max(m.shape[1] for m in all_masks_ordered)

        mask_atlas = jnp.concatenate([jnp.pad(m, ((0,0), (0, max_width - m.shape[1])), 'constant', constant_values=jr.TRANSPARENT_ID)
                                    for m in all_masks_ordered], axis=0)
        sprite_info_table = jnp.array(sprite_info, dtype=jnp.int32)
        
        # 5. Create the integer background raster
        background_raster = jnp.asarray(jr._create_id_mask(sprites_rgba["background"], color_to_id))

        return palette, mask_atlas, sprite_info_table, background_raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # 1. Draw dynamic geometry (walls) directly onto the background raster first
        # This is simpler than creating thousands of sprite commands for the walls
        #score_color_tuple = self.consts.SCORE_COLOR
        #score_id = self.color_to_id.get(score_color_tuple, 0) # Assuming COLOR_TO_ID is stored in __init__
        
        #base_raster = self.background_raster.at[self.consts.WALL_TOP_Y : self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT, :].set(score_id)
        #base_raster = base_raster.at[self.consts.WALL_BOTTOM_Y : self.consts.WALL_BOTTOM_Y + self.consts.WALL_BOTTOM_HEIGHT, :].set(score_id)

        # 2. Create an empty RenderPlan for the sprites
        plan = jr.create_initial_frame(max_sprites=32, sprite_info_table=self.sprite_info_table)

        # 3. Add sprite draw commands to the plan
        plan = jr.render_at(plan, self.consts.PLAYER_X, state.player_y, sprite_id=SPRITE_ID_PLAYER, depth_z=2)
        plan = jr.render_at(plan, self.consts.ENEMY_X, state.enemy_y, sprite_id=SPRITE_ID_ENEMY, depth_z=2)
        plan = jr.render_at(plan, state.ball_x, state.ball_y, sprite_id=SPRITE_ID_BALL, depth_z=3)

        # Add score commands to the plan
        player_digits = jr.int_to_digits(state.player_score, max_digits=2)
        enemy_digits = jr.int_to_digits(state.enemy_score, max_digits=2)
        plan = jr.render_label_selective(
            plan, x=120, y=3,
            all_digits=player_digits,
            num_to_render=2,
            char_sprite_ids=jnp.arange(SPRITE_ID_PLAYER_DIGITS_START, SPRITE_ID_PLAYER_DIGITS_START + 10),
            start_index=0,
            depth_z=3
        )
        plan = jr.render_label_selective(
            plan, x=10, y=3,
            all_digits=enemy_digits,
            num_to_render=2,
            char_sprite_ids=jnp.arange(SPRITE_ID_ENEMY_DIGITS_START, SPRITE_ID_ENEMY_DIGITS_START + 10),
            start_index=0,
            depth_z=3
        )

        # 4. Execute the plan to get the final object raster
        final_raster = jr.execute_plan_for_palette(
            plan,
            self.background_raster, # Use the raster with walls as the background
            self.mask_atlas,
            screen_height=210,
            screen_width=160,
            max_sprite_w=self.max_sprite_w,
            max_sprite_h=self.max_sprite_h
        )

        # 5. Perform the final palette lookup to get the color image
        return jr.render_from_palette(final_raster, self.palette) # Assumes render_from_palette is available