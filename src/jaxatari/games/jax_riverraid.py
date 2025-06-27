#import os
#from pty import spawn
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
DEFAULT_RIVER_WIDTH = 80


class RiverraidState(NamedTuple):
    turn_step: chex.Array
    master_key: chex.Array
    player_x: chex.Array
    river_left: chex.Array
    river_right: chex.Array
    river_inner_left: chex.Array
    river_inner_right: chex.Array
    river_state: chex.Array # 0 keep straight, 1 expanse, 2 shrinking
    river_island_present: chex.Array #0 no, 1 yes, 2 spawning, 3 removing, 4 removing helper
    alternation_cooldown: chex.Array
    river_alternation_length: chex.Array
    river_ongoing_alternation: chex.Array
    island_transition_state: chex.Array


class RiverraidInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


class RiverraidObservation(NamedTuple):
    player_x: chex.Array


# logic sperated into 3 branches: island, no_island, island_transition
# except for the transition phases, the states are managed at the top level
# islands are randomly spawned in no_island branch within expanse
# real game always spawns after expanse - straight - island, therefore island_transition_state
# islands are terminated when it randomly shrinks to 0 OR when the logic decides to remove the island
# then also shrink - straight - river shrink
@jax.jit
def update_river_banks(state: RiverraidState) -> RiverraidState:
    key = state.master_key
    key, alter_check_key, alter_select_key, alter_length_key = jax.random.split(key, 4)


    # Scroll the screen down
    scrolled_left = jnp.roll(state.river_left, 1)
    scrolled_right = jnp.roll(state.river_right, 1)
    scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
    scrolled_inner_right = jnp.roll(state.river_inner_right, 1)

    # jnp.roll puts the last element at the front
    # so we need to clear the history of the island
    scrolled_inner_left = scrolled_inner_left.at[0].set(-1)
    scrolled_inner_right = scrolled_inner_right.at[0].set(-1)
    state = state._replace(river_left=scrolled_left,
                           river_right=scrolled_right,
                           river_inner_left=scrolled_inner_left,
                           river_inner_right=scrolled_inner_right)

    # New state set here ONLY if no alternation is ongoing
    # later checked and change potentially REVERSED if cooldown is active
    # determine IF the river is to be altered
    new_river_state = jax.lax.cond(
        state.river_alternation_length <= 0,
        lambda state: (jax.random.choice(
            alter_select_key, jnp.array([0, 1, 2]), p=jnp.array([0.60, 0.20, 0.20])
        )),
        lambda state: state.river_state,
        operand=state
    )

    # alternation length is set here AND whenever an island spawns or despawns (due to the need for hard coding the segment)
    #jax.debug.print("Riverraid: new_river_state: {new_river_state}", new_river_state=new_river_state)
    new_alternation_length = jax.lax.cond(state.river_alternation_length <= 0,
                                          lambda state: jax.random.randint(alter_length_key, (), 1, 8),
                                          lambda state: state.river_alternation_length - 1,
                                          operand=state
                                          )
    # cooldown is handled and new_river state potentially reset to straight
    new_alternation_length, new_river_state, new_alternation_cooldown = jax.lax.cond(
        (state.alternation_cooldown > 0) & (state.river_alternation_length <= 0),
        lambda state: (jnp.array(0), jnp.array(0), state.alternation_cooldown - 1),
        lambda state: (new_alternation_length, new_river_state, new_alternation_length + 10),
        operand=state
    )

    state = state._replace( river_alternation_length=new_alternation_length,
                            river_state=new_river_state,
                            alternation_cooldown=new_alternation_cooldown)


    # used by both island and non island branches
    def straight(state: RiverraidState) -> RiverraidState:
        new_river_left = state.river_left.at[0].set(state.river_left[1])
        new_river_right = state.river_right.at[0].set(state.river_right[1])
        new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1])
        new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1])
        #jax.debug.print("straight state: {state}", state=state.river_state)
        return state._replace(river_left=new_river_left,
                              river_right=new_river_right,
                              river_inner_left=new_river_inner_left,
                              river_inner_right=new_river_inner_right)

    def no_island_branch(state: RiverraidState) -> RiverraidState:
        #jax.debug.print("NO ISLAND BRANCH")
        def expanse(state: RiverraidState) -> RiverraidState:
            new_river_left = state.river_left.at[0].set(state.river_left[1] - 3)
            new_river_right = state.river_right.at[0].set(state.river_right[1] + 3)

            key, subkey = jax.random.split(state.master_key)
            should_set_island = jax.random.bernoulli(subkey, 0.1)
            new_island_present = jax.lax.select(should_set_island, jnp.array(2), state.river_island_present)

            return state._replace(river_left=new_river_left,
                                  river_right=new_river_right,
                                  river_island_present=new_island_present)

        def shrink(state: RiverraidState) -> RiverraidState:
            new_river_left = state.river_left.at[0].set(state.river_left[1] + 3)
            new_river_right = state.river_right.at[0].set(state.river_right[1] - 3)
            return state._replace(river_left=new_river_left,
                                  river_right=new_river_right)

        return lax.switch(
            state.river_state,
            [straight, expanse, shrink],
            state
        )


    def island_branch(state: RiverraidState):
        #jax.debug.print("YES ISLAND BRANCH")
        def expanse(state: RiverraidState) -> RiverraidState:
            new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] - 3)
            new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] + 3)
            new_left = state.river_left.at[0].set(state.river_left[1])
            new_right = state.river_right.at[0].set(state.river_right[1])
            return state._replace(river_inner_left=new_river_inner_left,
                                  river_inner_right=new_river_inner_right,
                                  river_left=new_left,
                                  river_right=new_right)

        def shrink(state: RiverraidState) -> RiverraidState:
            proposed_inner_left = state.river_inner_left[1] + 3
            proposed_inner_right = state.river_inner_right[1] - 3
            new_river_inner_left, new_river_inner_right, new_island_present = jax.lax.cond(
                proposed_inner_left + 0 <= proposed_inner_right,
                lambda _: (
                    state.river_inner_left.at[0].set(proposed_inner_left),
                    state.river_inner_right.at[0].set(proposed_inner_right),
                    jnp.array(4)
                ),
                lambda _: (
                    state.river_inner_left.at[0].set(state.river_inner_left[1]),
                    state.river_inner_right.at[0].set(state.river_inner_right[1]),
                    state.river_island_present
                ),
                operand=None
            )

            new_left = state.river_left.at[0].set(state.river_left[1])
            new_right = state.river_right.at[0].set(state.river_right[1])


            #new_island_present = jax.lax.cond(new_river_inner_left[0] >= new_river_inner_right[0],
                                             # lambda _: jnp.array(0),
                                              #lambda state: state.river_island_present,
                                              #operand=state
                                              #)

            return state._replace(river_inner_left=new_river_inner_left,
                                  river_inner_right=new_river_inner_right,
                                  river_left=new_left,
                                  river_right=new_right,
                                  river_island_present=new_island_present
            )

        #jax.debug.print("YES ISLAND RIVER STATE {state}", state=state.river_state)
        return lax.switch(
            state.river_state,
            [straight, expanse, shrink],
            state
        )


    def island_transition(state: RiverraidState) -> RiverraidState:
        jax.debug.print("ISLAND TRANSITION BRANCH")
        def spawn_island(state: RiverraidState) -> RiverraidState:
            jax.debug.print("SPAWNING ISLAND")
            def straight(state: RiverraidState) -> RiverraidState:
                new_river_left = state.river_left.at[0].set(state.river_left[1])
                new_river_right = state.river_right.at[0].set(state.river_right[1])
                new_alternation_length, new_island_transition_state = (
                    jax.lax.switch(
                        state.island_transition_state,
                        [
                            lambda state: jax.lax.cond(
                                state.river_alternation_length <= 1,
                                lambda state: (jnp.array(10), jnp.array(1)),
                                lambda state: (state.river_alternation_length, state.island_transition_state),
                                operand=state
                            ),
                            lambda state: jax.lax.cond(
                                state.river_alternation_length <= 1,
                                lambda state: (state.river_alternation_length, jnp.array(2)),
                                lambda state: (state.river_alternation_length, state.island_transition_state),
                                operand=state
                            ),
                            lambda state: (state.river_alternation_length, state.island_transition_state)
                        ],
                        state
                    ))
                #jax.debug.print("straight vor ISLAND {state}", state=new_island_transition_state)
                return state._replace(river_left=new_river_left,
                                      river_right=new_river_right,
                                      river_alternation_length= new_alternation_length,
                                      island_transition_state=new_island_transition_state)

            def initiate_split(state: RiverraidState) -> RiverraidState:
                jax.debug.print("INIT SPLIIIIIIIIIIIIIIIIT")
                new_river_inner_left = state.river_inner_left.at[0].set(SCREEN_WIDTH // 2 - 1)
                new_river_inner_right = state.river_inner_right.at[0].set(SCREEN_WIDTH // 2 + 1)
                new_river_left = state.river_left.at[0].set(state.river_left[1])
                new_river_right = state.river_right.at[0].set(state.river_right[1])

                new_alternation_length = jnp.array(3)
                new_island_transition_state = jnp.array(0)
                new_river_state = jnp.array(1)
                new_island_present = jnp.array(1)

                return state._replace(river_inner_left=new_river_inner_left,
                                      river_inner_right=new_river_inner_right,
                                      river_left=new_river_left,
                                      river_right=new_river_right,
                                      river_alternation_length=new_alternation_length,
                                      island_transition_state=new_island_transition_state,
                                      river_state=new_river_state,
                                      river_island_present=new_island_present)

            return lax.switch(
                state.island_transition_state,
                [straight, straight, initiate_split],
                state
            )

        def terminate_island(state: RiverraidState) -> RiverraidState:
            def remove_island(state: RiverraidState) -> RiverraidState:
                jax.debug.print("REMOVING ISLAND")
                new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] + 3)
                new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] - 3)
                new_left = state.river_left.at[0].set(state.river_left[1])
                new_right = state.river_right.at[0].set(state.river_right[1])

                new_island_present, new_island_transition_state, new_alternation_length = jax.lax.cond(
                    new_river_inner_left[0] >= new_river_inner_right[0],
                    lambda _: (jnp.array(4), jnp.array(4), jnp.array(8)),
                    lambda state: (state.river_island_present, state.island_transition_state, jnp.array(1)),
                    operand=state
                )


                return state._replace(
                    river_inner_left=new_river_inner_left,
                    river_inner_right=new_river_inner_right,
                    river_left=new_left,
                    river_right=new_right,
                    river_island_present=new_island_present,
                    island_transition_state=new_island_transition_state,
                    river_alternation_length=new_alternation_length
                )

            def straight(state: RiverraidState) -> RiverraidState:
                jax.debug.print("STRAIGHT AFTER REMOVAL")
                new_river_left = state.river_left.at[0].set(state.river_left[1])
                new_river_right = state.river_right.at[0].set(state.river_right[1])

                jax.debug.print("alternation length: {length}", length=state.river_alternation_length)
                new_river_state, new_island_transition_state, new_island_present = (
                                                jax.lax.cond(state.river_alternation_length <= 1,
                                                lambda _: (jnp.array(2), jnp.array(0), jnp.array(0)),
                                                lambda state: (state.river_state, state.island_transition_state, state.river_island_present),
                                                operand=state
                                                ))

                new_alternation_length = jax.lax.cond(
                    new_island_present == 0,
                    lambda _: jax.random.randint(alter_length_key, (), 3, 8),
                    lambda _: state.river_alternation_length,
                    operand=None
                )

                return state._replace(river_left=new_river_left,
                                      river_right=new_river_right,
                                      island_transition_state=new_island_transition_state,
                                      river_alternation_length=new_alternation_length,
                                      river_state=new_river_state,
                                      river_island_present=new_island_present)

            jax.debug.print("YEEEEEEEEEEEEEEEEEEETING ISLAND")
            return jax.lax.cond(
                state.island_transition_state == 4,
                lambda state: straight(state),
                lambda state: remove_island(state),
                operand=state
            )

        return jax.lax.cond(jnp.logical_or(state.river_island_present == 0, state.river_island_present == 2),
                        lambda state: spawn_island(state),
                        lambda state: terminate_island(state),
                        operand=state)
        #return spawn_island(state)

    jax.debug.print("Riverraid: river_island_present: {island}", island=state.river_island_present)
    state = lax.switch(
        state.river_island_present,
        [no_island_branch, island_branch, island_transition, island_transition, island_transition],
        state
    )


    min_river_width = 15.0
    max_river_width = 130.0
    proposed_left = state.river_left[0]
    proposed_right = state.river_right[0]

    proposed_width = (proposed_right - proposed_left).astype(jnp.float32)
    proposed_center = (proposed_left + proposed_right).astype(jnp.float32) / 2.0

    clamped_width = lax.clamp(min_river_width, proposed_width, max_river_width)

    min_center = clamped_width / 2.0
    max_center = SCREEN_WIDTH - (clamped_width / 2.0)
    clamped_center = lax.clamp(min_center, proposed_center, max_center)

    final_left = jnp.round(clamped_center - clamped_width / 2.0).astype(jnp.int32)
    final_right = (final_left + jnp.round(clamped_width).astype(jnp.int32))

    new_left = state.river_left.at[0].set(final_left)
    new_right = state.river_right.at[0].set(final_right)

    state = state._replace(
        river_left=new_left,
        river_right=new_right,
        master_key=key
    )
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
        initial_key = jax.random.PRNGKey(1)

        state = RiverraidState(turn_step=0,
                               player_x=jnp.array(10),
                               river_left=jnp.full((SCREEN_HEIGHT,), river_start_x, dtype=jnp.int32),
                               river_right=jnp.full((SCREEN_HEIGHT,), river_end_x, dtype=jnp.int32),
                               river_inner_left=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               river_inner_right=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               river_state=jnp.array(0),
                               river_alternation_length=jnp.array(0),
                               master_key=initial_key,
                               river_ongoing_alternation=jnp.array(0),
                               river_island_present=jnp.array(0),
                               alternation_cooldown=jnp.array(10),
                               island_transition_state=jnp.array(0)
                               )
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

    initial_key = jax.random.PRNGKey(4)
    initial_observation, state = jitted_reset()
    state = state._replace(master_key=initial_key)

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