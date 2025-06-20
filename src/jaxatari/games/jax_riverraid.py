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
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 200
DEFAULT_RIVER_WIDTH = 90


class RiverraidState(NamedTuple):
    turn_step: chex.Array
    player_x: chex.Array
    river_left: chex.Array
    river_right: chex.Array
    river_inner_left: chex.Array
    river_inner_right: chex.Array
    river_state: chex.Array # 0 keep straight, 1 expanse, 2 shrinking, 3 splitting, 4 splitted expanse, 5 splitted shrinking, 6 splitted straight, 7 terminate island
    river_alternation_length: chex.Array


class RiverraidInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


class RiverraidObservation(NamedTuple):
    player_x: chex.Array


#@jax.jit
def update_river_banks(state: RiverraidState) -> RiverraidState:
    def select_alternation(state: RiverraidState):
        key = jax.random.PRNGKey(state.turn_step)
        key, subkey = jax.random.split(key)
        new_river_state = jax.random.randint(key, (), 1, 6)
        alternation_length = jax.random.randint(subkey, (), 4, 9)
        return state._replace(river_state=new_river_state,
                              river_alternation_length=alternation_length)

    def alter_expanse(state: RiverraidState) -> RiverraidState:
        new_top_left = state.river_left[1] - 3
        new_top_right = state.river_right[1] + 3

        new_river_left = state.river_left.at[0].set(new_top_left)
        new_river_right = state.river_right.at[0].set(new_top_right)

        new_alter_length = state.river_alternation_length - 1
        return state._replace(river_left=new_river_left,
                              river_right=new_river_right,
                              river_alternation_length=new_alter_length)

    def alter_shrink(state: RiverraidState) -> RiverraidState:
        new_top_left = state.river_left[1] + 3
        new_top_right = state.river_right[1] - 3

        new_river_left = state.river_left.at[0].set(new_top_left)
        new_river_right = state.river_right.at[0].set(new_top_right)

        new_alter_length = state.river_alternation_length - 1

        return state._replace(river_left=new_river_left,
                              river_right=new_river_right,
                              river_alternation_length=new_alter_length)

    def alter_split(state: RiverraidState) -> RiverraidState:
        def split(state: RiverraidState) -> RiverraidState:
            def initiate_split(state: RiverraidState) -> RiverraidState:
                new_river_inner_left = state.river_inner_left.at[0].set(SCREEN_WIDTH // 2 - 1)
                new_river_inner_right = state.river_inner_right.at[0].set(SCREEN_WIDTH // 2 + 1)
                return state._replace(river_inner_left=new_river_inner_left,
                                      river_inner_right=new_river_inner_right,
                                      river_state=jnp.array(3))

            def continue_split(state: RiverraidState) -> RiverraidState:
                jax.debug.print("river left 0: {value}", value=state.river_inner_left[0])
                jax.debug.print("river left 1: {value}", value=state.river_inner_left[1])
                new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] - 3)
                new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] + 3)
                new_alter_length = state.river_alternation_length - 1
                new_river_state = jax.lax.cond(new_alter_length == 0,
                                               lambda: jnp.array(6),
                                               lambda: state.river_state)
                return state._replace(river_inner_left=new_river_inner_left,
                                      river_inner_right=new_river_inner_right,
                                      river_alternation_length=new_alter_length,
                                      river_state=new_river_state)


            return jax.lax.cond(state.river_inner_left[1] < 0,
                         lambda state: initiate_split(state),
                         lambda state: continue_split(state),
                         operand=state)

        return jax.lax.cond(
            True,
            lambda state: split(state),
            lambda state: state,
            operand=state
        )

    def alter_splitted_expanse(state: RiverraidState) -> RiverraidState:
        return alter_shrink(state)

    def alter_splitted_shrink(state: RiverraidState) -> RiverraidState:
        return alter_expanse(state)

    def continue_straight(state: RiverraidState) -> RiverraidState:
        return jax.lax.cond(
            (state.river_state == 0) | (state.river_state == 6),
            lambda state: state._replace(
                river_left=state.river_left.at[0].set(state.river_left[1]),
                river_right=state.river_right.at[0].set(state.river_right[1]),
                river_inner_left=state.river_inner_left.at[0].set(state.river_inner_left[1]),
                river_inner_right=state.river_inner_right.at[0].set(state.river_inner_right[1])
            ),
            lambda state: state,
            state
        )

    # Scroll the screen down
    scrolled_left = jnp.roll(state.river_left, 1)
    scrolled_right = jnp.roll(state.river_right, 1)
    scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
    scrolled_inner_right = jnp.roll(state.river_inner_right, 1)
    state = state._replace(river_left=scrolled_left,
                   river_right=scrolled_right,
                   river_inner_left=scrolled_inner_left,
                   river_inner_right=scrolled_inner_right)


    # determine IF the river is to be altered
    key = jax.random.PRNGKey(state.turn_step)
    alter_river = jax.random.bernoulli(key, p=0.05)

    # If the river is to be altered, select the type of alteration
    state = jax.lax.cond(alter_river & (state.river_alternation_length <= 0),
                         lambda state: select_alternation(state),
                         lambda state: continue_straight(state),
                         operand=state)


    # If an alternation is in progress, continue with the alteration
    state = jax.lax.cond(
        state.river_state > 0,
        lambda state: lax.switch(
            state.river_state - 1,
            [
                lambda state: alter_expanse(state),
                lambda state: alter_shrink(state),
                lambda state: alter_split(state),
                lambda state: alter_splitted_expanse(state),
                lambda state: alter_splitted_shrink(state),
            ],
            state,
        ),
        lambda state: state,
        operand=state,
    )

    new_river_state = jax.lax.cond(
        state.river_alternation_length <= 0,
        lambda: jnp.array(0),
        lambda: state.river_state
    )
    state = state._replace(river_state=new_river_state)


    return state



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
        river_start_x = (SCREEN_WIDTH - DEFAULT_RIVER_WIDTH) // 2
        river_end_x = river_start_x + DEFAULT_RIVER_WIDTH

        state = RiverraidState(turn_step=0,
                               player_x=jnp.array(10),
                               river_left=jnp.full((SCREEN_HEIGHT,), river_start_x, dtype=jnp.int32),
                               river_right=jnp.full((SCREEN_HEIGHT,), river_end_x, dtype=jnp.int32),
                               river_inner_left=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               river_inner_right=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               river_state=jnp.array(0),
                               river_alternation_length=jnp.array(0))
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def step(self, state: RiverraidState, action: Action) -> Tuple[RiverraidObservation, RiverraidState, RiverraidInfo]:
        new_state = state._replace(turn_step=state.turn_step + 1)
        new_state = update_river_banks(new_state)
        #jax.debug.print("left: {new_state}", new_state=new_state.river_inner_left)

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
        green_banks = jnp.array([26, 132, 26], dtype=jnp.uint8)
        blue_river = jnp.array([42, 42, 189], dtype=jnp.uint8)

        left_banks = state.river_left[:, None]
        right_banks = state.river_right[:, None]

        inner_left_banks = state.river_inner_left[:, None]
        inner_right_banks = state.river_inner_right[:, None]

        x_coords = jnp.arange(SCREEN_WIDTH)
        is_river =  (x_coords > left_banks) & (x_coords < right_banks) & jnp.logical_or(x_coords < inner_left_banks, x_coords > inner_right_banks)

        # The raster is  (HEIGHT, WIDTH, 3)
        raster = jnp.where(is_river[..., None], blue_river, green_banks)

        # transpose it to (WIDTH, HEIGHT, 3)
        return jnp.transpose(raster, (1, 0, 2))


if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 24)

    game = JaxRiverraid(frameskip=1)
    renderer = RiverraidRenderer()

    jitted_reset = jax.jit(game.reset)
    jitted_step = jax.jit(game.step)
    jitted_render = jax.jit(renderer.render)

    initial_observation, state = jitted_reset()


    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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
        aj.update_pygame(screen, render_output, 1, SCREEN_WIDTH, SCREEN_HEIGHT)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()