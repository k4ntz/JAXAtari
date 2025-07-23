import os
import chex
import jax.lax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from jax import random as jrandom
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs, EnvState
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces

class BoxingConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    initial_x_white: int = 30
    initial_y_white: int = 35
    initial_x_black: int = 115
    initial_y_black: int = 130
    x_boundaries = (28, 120)
    y_boundaries = (30, 135)
class BoxingState(NamedTuple):
    player_score: chex.Array
    enemy_score: chex.Array
    player_x: chex.Array
    player_y: chex.Array
    time: chex.Array


class EntityPositions(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray

class BoxingObservation(NamedTuple):
    player: EntityPositions
class BoxingInfo(NamedTuple):
    time: jnp.ndarray

class JaxBoxing(JaxEnvironment[BoxingState, BoxingObservation, BoxingInfo, BoxingConstants]):
    def __init__(self, consts: BoxingConstants = None, reward_funcs: list[callable] = None):
        consts = consts or BoxingConstants()
        super().__init__(consts)
        self.renderer = BoxingRenderer(consts)
        if reward_funcs is not None:
            self.reward_funcs = reward_funcs
        self.action_set = [
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UP,
            Action.UPRIGHT,
            Action.DOWNRIGHT,
            Action.UPLEFT,
            Action.DOWNLEFT,
        ]

    def reset(self, key=None) -> Tuple[BoxingObservation, BoxingState]:
        state = BoxingState(
            player_score=jnp.array(9),
            enemy_score=jnp.array(0),
            player_x=self.consts.initial_x_white,
            player_y=self.consts.initial_y_white,
            time= jnp.array(7200)
        )

        initial_obs = self._get_observation(state)
        return initial_obs, state

    def player_step(self, state, action: chex.Array):
        # Movement flags
        left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)
        up = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT)
        down = (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT)

        # Apply movement
        delta_x = jnp.where(right, 2, 0) - jnp.where(left, 2, 0)
        delta_y = jnp.where(down, 2, 0) - jnp.where(up, 2, 0)

        player_x = jnp.clip(state.player_x + delta_x, self.consts.x_boundaries[0], self.consts.x_boundaries[1])
        player_y = jnp.clip(state.player_y + delta_y, self.consts.y_boundaries[0], self.consts.y_boundaries[1])

        return state._replace(player_x=player_x, player_y=player_y)
    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, float, bool, BoxingInfo]:
        state = self.player_step(state, action)
        new_time = jnp.maximum(0, state.time - 1)
        new_state = BoxingState(player_score=state.player_score,
                                enemy_score=state.enemy_score,
                                player_x=state.player_x,
                                player_y=state.player_y,
                                time = new_time
                                )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(new_state, action)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)
        return observation, new_state, env_reward, done, info

    def render(self, state: BoxingState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_done(self, state: EnvState) -> bool:
        return False
    def _get_reward(self, previous_state: EnvState, state: EnvState) -> float:
        return 0.0
    def _get_observation(self, state: BoxingState):
        return BoxingObservation(
            player=EntityPositions(
                x=jnp.array([0.0]),  # Placeholder for player x position
                y=jnp.array([0.0])   # Placeholder for player y position
            )
        )
    def _get_info(self, state: BoxingState, all_rewards: jnp.ndarray) -> BoxingInfo:
        return BoxingInfo(
            time=state.time
        )
    def _get_all_rewards(self, state: BoxingState, action: chex.Array) -> jnp.ndarray:
        return jnp.array([0.0])
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

class BoxingRenderer(JAXGameRenderer):
    def __init__(self, consts: BoxingConstants = None):
        self.consts = consts or BoxingConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            self.DIGITS,
            self.TIME_SEPERATION,
            self.ENEMY_SCORE_DIGITS,
        ) = self.load_sprites()
    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        background = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/background.npy"))
        player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/white_player.npy"))
        enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/black_player.npy"))

        DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/boxing/digits_time/{}.npy"))
        ENEMY_SCORE_DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/boxing/digits_black/{}.npy"))
        TIME_SEPERATION = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/digits_time_seperation.npy"))


        SPRITE_BG = jnp.expand_dims(background, axis=0)

        return (SPRITE_BG, player, enemy, DIGITS, TIME_SEPERATION, ENEMY_SCORE_DIGITS)
    def render(self, state):
        def render_time(self, raster, time):
            total_seconds = time // 60
            minutes = (total_seconds // 60).astype(int)
            seconds = (total_seconds % 60).astype(int)

            # Get digits
            min_tens = minutes // 10
            min_ones = minutes % 10
            sec_tens = seconds // 10
            sec_ones = seconds % 10

            # Render digits using your digit sprites at the desired location
            raster = jr.render_at(raster, 60, 15, self.DIGITS[min_tens])
            raster = jr.render_at(raster, 68, 15, self.DIGITS[min_ones])
            raster = jr.render_at(raster, 88, 15, self.DIGITS[sec_tens])
            raster = jr.render_at(raster, 96, 15, self.DIGITS[sec_ones])
            return raster

        def render_score(self, raster, score, x, y, digits):
            tens = score // 10
            ones = score % 10
            raster = jnp.where(score >= 10,jr.render_at(raster, x, y, digits[tens]), raster)
            raster = jr.render_at(raster, x+3, y, digits[ones])
            return raster


        raster = jr.create_initial_frame(width=160, height=210)
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0,0, frame_bg)
        raster = jr.render_at(raster, 80, 15, self.TIME_SEPERATION)
        raster = jr.render_at(raster, state.player_x, state.player_y, self.SPRITE_PLAYER)
        raster = jr.render_at(raster, self.consts.initial_x_black, self.consts.initial_y_black, self.SPRITE_ENEMY)
        raster = render_time(self, raster, state.time)
        raster = render_score(self,raster, state.player_score, 45, 7, self.ENEMY_SCORE_DIGITS)
        raster = render_score(self, raster, state.enemy_score, 104, 7, self.ENEMY_SCORE_DIGITS)
        return raster

