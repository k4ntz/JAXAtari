import os
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax

from gymnax.environments import spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.games.jax_pong import WIDTH
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj


# -------- Game constants --------



class RiverraidState(NamedTuple):
    turn_step: chex.Array
    player_x: chex.Array
    river_left: chex.Array
    river_right: chex.Array


class RiverraidInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


class RiverraidObservation(NamedTuple):
    player_x: chex.Array


class JaxRiverraid(JaxEnvironment):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable] = None):
        super().__init__()
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE
        }
        self.obs_size = 420

    def _get_observation(self, state: RiverraidState) -> RiverraidObservation:
        observation = RiverraidObservation(player_x=state.player_x)
        return observation

    def reset(self, key=None) -> Tuple[RiverraidObservation, RiverraidState]:
        state = RiverraidState(turn_step=0,
                               player_x=jnp.array(10),
                               river_left=jnp.array(50),
                               river_right=jnp.array(50))
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def step(self, state: RiverraidState, action: Action) -> Tuple[RiverraidObservation, RiverraidState, RiverraidInfo]:
        new_state = state._replace(turn_step=state.turn_step + 1)

        observation = self._get_observation(new_state)
        reward = self._get_env_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state, jnp.zeros(1))

        return observation, new_state, reward, done, info

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: RiverraidState, state: RiverraidState):
        return 420

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: RiverraidState, state: RiverraidState) -> chex.Array:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: RiverraidState) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: RiverraidState, all_rewards: chex.Array) -> RiverraidState:
        return RiverraidInfo(time=state.turn_step, all_rewards=all_rewards)


class RiverraidRenderer(AtraJaxisRenderer):
    def render(self, state: RiverraidState):
        game_width = 160
        game_height = 210

        green_banks = jnp.array([26, 132, 26], dtype=jnp.uint8)
        blue_river = jnp.array([42, 42, 189], dtype=jnp.uint8)

        raster = jnp.full((game_height, game_width, 3), green_banks, dtype=jnp.uint8)

        river_start = state.river_left
        river_end = WIDTH - state.river_right

        raster = raster.at[river_start:river_end, :].set(blue_river)

        return raster


if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 24)

    game = JaxRiverraid(frameskip=1)
    renderer = RiverraidRenderer()

    jitted_reset = jax.jit(game.reset)
    jitted_step = jax.jit(game.step)
    jitted_render = jax.jit(renderer.render)

    initial_observation, state = jitted_reset()


    screen = pygame.display.set_mode((160 * 3, 210 * 3))
    pygame.display.set_caption("Riverraid")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = Action.NOOP
        observation, state, reward, done, info = jitted_step(state, action)

        render_output = jitted_render(state)
        aj.update_pygame(screen, render_output, 3, 160, 210)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()