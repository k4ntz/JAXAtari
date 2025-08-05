import os
#from pty import spawn
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax

from gymnax.environments import spaces
#from ocatari.vision.skiing import player_c

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj


# -------- Game constants --------
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 200
DEFAULT_RIVER_WIDTH = 80
MIN_RIVER_WIDTH = 30
MAX_RIVER_WIDTH = 130
MAX_ENEMIES = 10
MINIMUM_SPAWN_COOLDOWN = 20


class RiverraidState(NamedTuple):
    turn_step: chex.Array
    master_key: chex.Array
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

    segment_state : chex.Array
    segment_transition_state: chex.Array
    segment_straigt_counter: chex.Array
    dam_position: chex.Array

    player_x: chex.Array
    player_y: chex.Array
    player_velocity: chex.Array
    player_direction: chex.Array  # 0 left, 1 straight, 2 right
    player_state: chex.Array

    player_bullet_x: chex.Array
    player_bullet_y: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_state: chex.Array  # 0 empty/dead, 1 alive
    enemy_direction: chex.Array  # 0 left static, 1 right static, 2 left moving, 3 right moving
    spawn_cooldown: chex.Array



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
def generate_altering_river(state: RiverraidState) -> RiverraidState:
    jax.debug.print("ALTERING RIVER IS BEING CALLED AAA")
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
            should_set_island = jnp.logical_and(jax.random.bernoulli(subkey, 0.1),
                                                new_river_right[0] - new_river_left[0] > 50)
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
            #jax.debug.print("EXPANDING ISLAND")
            new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] - 3)
            new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] + 3)
            new_left = state.river_left.at[0].set(state.river_left[1])
            new_right = state.river_right.at[0].set(state.river_right[1])
            return state._replace(river_inner_left=new_river_inner_left,
                                  river_inner_right=new_river_inner_right,
                                  river_left=new_left,
                                  river_right=new_right)

        def shrink(state: RiverraidState) -> RiverraidState:
            #jax.debug.print("SHRINKING ISLAND")
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
        #jax.debug.print("ISLAND TRANSITION BRANCH")
        def spawn_island(state: RiverraidState) -> RiverraidState:
            #jax.debug.print("SPAWNING ISLAND")
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
                #jax.debug.print("INIT SPLIIIIIIIIIIIIIIIIT")
                new_river_inner_left = state.river_inner_left.at[0].set(SCREEN_WIDTH // 2 - 1)
                new_river_inner_right = state.river_inner_right.at[0].set(SCREEN_WIDTH // 2 + 1)
                new_river_left = state.river_left.at[0].set(state.river_left[1])
                new_river_right = state.river_right.at[0].set(state.river_right[1])

                new_alternation_length = jax.random.randint(alter_length_key, (), 4, 12)
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
                #jax.debug.print("REMOVING ISLAND")
                new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] + 3)
                new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] - 3)
                new_left = state.river_left.at[0].set(state.river_left[1])
                new_right = state.river_right.at[0].set(state.river_right[1])

                new_island_present, new_island_transition_state, new_alternation_length = jax.lax.cond(
                    new_river_inner_left[0] >= new_river_inner_right[0],
                    lambda _: (jnp.array(4), jnp.array(4), jnp.array(14)),
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
                #jax.debug.print("STRAIGHT AFTER REMOVAL")
                new_river_left = state.river_left.at[0].set(state.river_left[1])
                new_river_right = state.river_right.at[0].set(state.river_right[1])

                #jax.debug.print("alternation length: {length}", length=state.river_alternation_length)
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

            #jax.debug.print("YEEEEEEEEEEEEEEEEEEETING ISLAND")
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

   # jax.debug.print("Riverraid: river_island_present: {island}", island=state.river_island_present)
    new_river_island_present = jax.lax.cond(
        jnp.logical_and(
            state.river_island_present >= 2,
            jnp.logical_and(
                state.river_inner_left[0] <= state.river_inner_right[0],
                state.river_right[1] - state.river_left[1] < MIN_RIVER_WIDTH * 2 + 20
            )
        ),
        lambda state: jnp.array(0),
        lambda state: state.river_island_present,
        operand=state
    )
    state = state._replace(river_island_present=new_river_island_present)

    state = jax.lax.cond(
        state.river_island_present == 0,
        lambda state: no_island_branch(state),
        lambda state: jax.lax.cond(
            state.river_island_present == 1,
            lambda state: island_branch(state),
            lambda state: island_transition(state),
            operand=state
        ),
        operand=state
    )

    def no_island_clamping(state: RiverraidState) -> RiverraidState:
        min_river_width = MIN_RIVER_WIDTH
        max_river_width = MAX_RIVER_WIDTH
        proposed_left = state.river_left[0]
        proposed_right = state.river_right[0]

        proposed_width = (proposed_right - proposed_left).astype(jnp.float32)
        proposed_center = (proposed_left + proposed_right).astype(jnp.float32) / 2.0

        clamped_width = lax.clamp(min_river_width, proposed_width.astype(jnp.int32), max_river_width)

        min_center = clamped_width / 2.0
        max_center = SCREEN_WIDTH - (clamped_width / 2.0)
        clamped_center = lax.clamp(min_center, proposed_center, max_center)

        final_left = jnp.round(clamped_center - clamped_width / 2.0).astype(jnp.int32)
        final_right = (final_left + jnp.round(clamped_width).astype(jnp.int32))

        new_left = state.river_left.at[0].set(final_left)
        new_right = state.river_right.at[0].set(final_right)

        new_alternation_length, new_alternation_cooldown =jax.lax.cond(jnp.logical_and(final_left != proposed_left, state.river_state == 2),
                                             lambda _: (jnp.array(0), jnp.array(0)),
                                             lambda state: (state.river_alternation_length, state.alternation_cooldown),
                                             operand=state
                                            )

        return state._replace(
            river_left=new_left,
            river_right=new_right,
            river_alternation_length=new_alternation_length,
            alternation_cooldown=new_alternation_cooldown
        )

    def yes_island_clamping(state: RiverraidState) -> RiverraidState:
        outer_left = state.river_left[0]
        outer_right = state.river_right[0]
        proposed_inner_left = state.river_inner_left[0]
        proposed_inner_right = state.river_inner_right[0]
        min_river_width = MIN_RIVER_WIDTH

        new_inner_left = jax.lax.cond(
            (proposed_inner_left - outer_left).astype(jnp.float32) < min_river_width,
            lambda state: state.river_inner_left.at[0].set((outer_left + min_river_width).astype(jnp.int32)),
            lambda state: state.river_inner_left,
            operand=state
        )

        new_inner_right = jax.lax.cond(
            (outer_right - proposed_inner_right).astype(jnp.float32) < min_river_width,
            lambda state: state.river_inner_right.at[0].set((outer_right - min_river_width).astype(jnp.int32)),
            lambda state: state.river_inner_right,
            operand=state
        )

        new_alternation_length, new_alternation_cooldown =jax.lax.cond(jnp.logical_and(proposed_inner_left != new_inner_left[0], state.river_state == 1),
                                             lambda _: (jnp.array(0), jnp.array(0)),
                                             lambda state: (state.river_alternation_length, state.alternation_cooldown),
                                             operand=state
                                            )
        return state._replace(
            river_inner_left=new_inner_left,
            river_inner_right=new_inner_right,
            river_alternation_length=new_alternation_length,
            alternation_cooldown=new_alternation_cooldown
        )

    state = jax.lax.cond(state.river_island_present == 0,
                        lambda state: no_island_clamping(state),
                        lambda state: yes_island_clamping(state),
                        operand=state
                        )
    return state._replace(master_key=key)

@jax.jit
def generate_straight_river(state: RiverraidState) -> RiverraidState:
    scrolled_left = jnp.roll(state.river_left, 1)
    scrolled_right = jnp.roll(state.river_right, 1)
    scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
    scrolled_inner_right = jnp.roll(state.river_inner_right, 1)
    scrolled_inner_left = scrolled_inner_left.at[0].set(-1)
    scrolled_inner_right = scrolled_inner_right.at[0].set(-1)

    current_width = scrolled_right[1] - scrolled_left[1]
    should_expand = current_width < MIN_RIVER_WIDTH * 2

    new_top_left = jnp.where(should_expand, scrolled_left[1] - 3, scrolled_left[1])
    new_top_right = jnp.where(should_expand, scrolled_right[1] + 3, scrolled_right[1])
    new_river_left = scrolled_left.at[0].set(new_top_left)
    new_river_right = scrolled_right.at[0].set(new_top_right)

    return state._replace(
        river_left=new_river_left,
        river_right=new_river_right,
        river_inner_left=scrolled_inner_left,
        river_inner_right=scrolled_inner_right,
        river_state=jnp.array(0)
    )

@jax.jit
def generate_segment_transition(state: RiverraidState) -> RiverraidState:
    #jax.debug.print("TRANSITIONING SEGMENT")
    def scroll_empty_island(state: RiverraidState) -> RiverraidState:
        scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
        scrolled_inner_right = jnp.roll(state.river_inner_right, 1)
        scrolled_inner_left = scrolled_inner_left.at[0].set(-1)
        scrolled_inner_right = scrolled_inner_right.at[0].set(-1)
        return state._replace(river_inner_left=scrolled_inner_left,
                                river_inner_right=scrolled_inner_right)

    def first_call(state: RiverraidState) -> RiverraidState:
        new_island_present, new_segment_transition_state = jax.lax.cond(state.river_island_present == 0,
                                        lambda state: (jnp.array(0), jnp.array(2)),
                                        lambda state: (jnp.array(3), jnp.array(1)),
                                        operand=state)
        return generate_altering_river(state._replace(river_island_present=new_island_present,
                                                      segment_transition_state=new_segment_transition_state))

    def remove_island(state: RiverraidState) -> RiverraidState:
        new_state = generate_altering_river(state)
        new_segment_transition_state = jax.lax.cond(jnp.logical_and(new_state.river_island_present == 0, new_state.river_state == 0),
                                                    lambda state: jnp.array(2),
                                                    lambda state: state.segment_transition_state,
                                                    operand=state)
        return new_state._replace(segment_transition_state=new_segment_transition_state)

    def shrink_to_damsize(state: RiverraidState) -> RiverraidState:
        scrolled_left = jnp.roll(state.river_left, 1)
        scrolled_right = jnp.roll(state.river_right, 1)
        #jax.debug.print("SHRINKING TO DAM SIZE")

        new_river_left = scrolled_left.at[0].set(scrolled_left[1] + 3)
        new_river_right = scrolled_right.at[0].set(scrolled_right[1] - 3)

        new_segment_transition_state = jax.lax.cond(new_river_right[0] - new_river_left[0] <= MIN_RIVER_WIDTH + 12,
                                                    lambda state: jnp.array(3),
                                                    lambda state: state.segment_transition_state,
                                                    operand=state
                                                    )
        new_state = scroll_empty_island(state)
        return new_state._replace(river_left=new_river_left,
                                river_right=new_river_right,
                                segment_transition_state=new_segment_transition_state)

    def straight_until_dam(state: RiverraidState) -> RiverraidState:
        scrolled_left = jnp.roll(state.river_left, 1)
        scrolled_right = jnp.roll(state.river_right, 1)
        new_river_left = scrolled_left.at[0].set(state.river_left[0])
        new_river_right = scrolled_right.at[0].set(state.river_right[0])
        new_segment_straight_counter = state.segment_straigt_counter - 1
        new_transition_state = jax.lax.cond(new_segment_straight_counter <= 0,
                                                    lambda state: jnp.array(4),
                                                    lambda state: state.segment_transition_state,
                                                    operand=state)
        new_state = scroll_empty_island(state)
        return new_state._replace(river_left=new_river_left,
                              river_right=new_river_right,
                              segment_straigt_counter=new_segment_straight_counter,
                              segment_transition_state=new_transition_state)

    def dam_into_new_segment(state: RiverraidState) -> RiverraidState:
        #jax.debug.print("SETTING DAM")
        #scrolled_left = jnp.roll(state.river_left, 1)
        #scrolled_right = jnp.roll(state.river_right, 1)
        new_state = scroll_empty_island(state)
        new_segment_straight_counter = jnp.array(8)
        dam_position = state.dam_position.at[0].set(1)

        new_river_state = jnp.array(0)
        new_alternation_length = jnp.array(10)
        new_alternation_cooldown = jnp.array(0)
        new_segment_state = state.segment_state + 1
        new_segment_transition_state = jnp.array(0)
        return new_state._replace(segment_state=new_segment_state,
                              segment_transition_state=new_segment_transition_state,
                              dam_position=dam_position,
                              segment_straigt_counter=new_segment_straight_counter,
                              river_state=new_river_state,
                              river_alternation_length=new_alternation_length,
                              alternation_cooldown=new_alternation_cooldown,
                              #river_left=scrolled_left.at[0].set(scrolled_left[1]),
                              #river_right=scrolled_right.at[0].set(scrolled_right[1]))
                                  )


    #jax.debug.print("Riverraid: segment_transition_state: {segment_transition_state}", segment_transition_state=state.segment_transition_state)
    return jax.lax.switch(state.segment_transition_state,[first_call,
                                                                remove_island,
                                                                shrink_to_damsize,
                                                                straight_until_dam,
                                                                dam_into_new_segment],
                                                               operand=state)


@jax.jit
def update_river_banks(state: RiverraidState) -> RiverraidState:
    new_segment_state = jax.lax.cond(
        (state.turn_step % 400) == 0,
        lambda state: state.segment_state + 1,
        lambda state: state.segment_state,
        operand=state
    )
    jax.debug.print("Riverraid: segment_state: {segment_state}", segment_state=new_segment_state)
    state = state._replace(segment_state=new_segment_state % 4)
    return jax.lax.switch(state.segment_state, [lambda state: generate_altering_river(state),
                                                lambda state: generate_segment_transition(state),
                                                lambda state: generate_straight_river(state),
                                                lambda state: generate_segment_transition(state)],
                                                operand=state)

@jax.jit
def roll_static_objects(state: RiverraidState) -> RiverraidState:
    new_dam_position = jnp.roll(state.dam_position, 1)
    #last index to 0 to prevent rolling over
    new_dam_position = new_dam_position.at[-1].set(0)
    return state._replace(dam_position=new_dam_position)

def player_movement(state: RiverraidState, action: Action) -> RiverraidState:
    press_right = jnp.any(
        jnp.array([action == Action.RIGHT, action == Action.RIGHTFIRE])
    )

    press_left = jnp.any(
        jnp.array([action == Action.LEFT, action == Action.LEFTFIRE])
    )

    new_velocity = jax.lax.cond(
        (press_left == 0) & (press_right == 0),
        lambda state: jnp.array(0, dtype=state.player_velocity.dtype),
        lambda state: state.player_velocity + (press_right * 0.1) - (press_left * 0.1),
        operand=state
    )

    new_velocity = jax.lax.cond(press_right == 0,
        lambda state: jnp.clip(new_velocity, -3, 0),
        lambda state: jnp.clip(new_velocity, 0, 3),
        operand=state
    )

    new_x = state.player_x + new_velocity

    # check collision with river banks -> invoke death
    player_state = jax.lax.cond(jnp.logical_or(new_x <= state.river_left[SCREEN_HEIGHT - 30] + 1, new_x >= state.river_right[SCREEN_HEIGHT - 30] - 8),
                 lambda state: jnp.array(1),
                 lambda state: state.player_state,
                 operand=state)

    # check collision with island -> invoke death
    player_state = jax.lax.cond(
        jnp.logical_and(
            state.river_inner_left[SCREEN_HEIGHT - 30] >= 0,
            jnp.logical_and(
                new_x >= state.river_inner_left[SCREEN_HEIGHT - 30] - 8,
                new_x <= state.river_inner_right[SCREEN_HEIGHT - 30] + 1
            )
        ),
        lambda state: jnp.array(1),
        lambda state: player_state,
        operand=state
    )

    return state._replace(player_x=new_x,
                          player_velocity=new_velocity,
                          player_state=player_state)

@jax.jit
def get_action_from_keyboard(state: RiverraidState) -> Action:
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    shooting = keys[pygame.K_SPACE]

    left_only = left and not right
    right_only = right and not left

    if shooting:
        if left_only:
            return Action.LEFTFIRE
        elif right_only:
            return Action.RIGHTFIRE
        else:
            return Action.FIRE
    else:
        if left_only:
            return Action.LEFT
        elif right_only:
            return Action.RIGHT
        else:
            return Action.NOOP


def player_shooting(state, action):
    shooting = jnp.any(
        jnp.array([action == Action.LEFTFIRE, action == Action.RIGHTFIRE, action == Action.FIRE])
    )
    new_bullet_x, new_bullet_y = jax.lax.cond(
        jnp.logical_and(
            shooting,
            state.player_bullet_y < 0),
        lambda state: ((state.player_x + 2).astype(jnp.float32), (state.player_y - 0).astype(jnp.float32)),
        lambda state: (state.player_bullet_x.astype(jnp.float32), (state.player_bullet_y - 5).astype(jnp.float32)),
        operand=state
    )

    # collision with outer banks
    new_bullet_x, new_bullet_y = jax.lax.cond(
        jnp.logical_or(new_bullet_x <= state.river_left[new_bullet_y.astype(jnp.int32)],
                        new_bullet_x >= state.river_right[new_bullet_y.astype(jnp.int32)]),
        lambda state: (jnp.array(-1, dtype=jnp.float32), jnp.array(-1, dtype=jnp.float32)),
        lambda state: (new_bullet_x, new_bullet_y),
        operand=state
    )

    # collision with island
    new_bullet_x, new_bullet_y = jax.lax.cond(
        jnp.logical_and(new_bullet_x >= state.river_inner_left[new_bullet_y.astype(jnp.int32)],
                        new_bullet_x <= state.river_inner_right[new_bullet_y.astype(jnp.int32)]),
        lambda state: (jnp.array(-1, dtype=jnp.float32), jnp.array(-1, dtype=jnp.float32)),
        lambda state: (new_bullet_x, new_bullet_y),
        operand=state
    )


    jax.debug.print("Riverraid: player_bullet_x: {bullet_x}, player_bullet_y: {bullet_y}", bullet_x=new_bullet_x, bullet_y=new_bullet_y)
    return state._replace(player_bullet_x=new_bullet_x,
                              player_bullet_y=new_bullet_y)

# TODO cooldown between spawns
@jax.jit
def spawn_enemies(state):
    jax.debug.print("SPAWNING ENEMY")
    key, spawn_key, x_key = jax.random.split(state.master_key, 3)
    #spawn_enemy = jax.random.bernoulli(spawn_key, 0.07)

    # 0 boat, 1 helicopter, 2 plane
    new_single_enemy_type = jax.random.randint(spawn_key, (), 0, 3)
    free_enemy_idx = jax.lax.cond(
        jnp.any(state.enemy_state == 0),
        lambda state: jnp.argmax(state.enemy_state == 0),
        lambda _: jnp.array(-1, dtype=jnp.int32),
        operand=state
    )

    new_enemy_type = state.enemy_type.at[free_enemy_idx].set(new_single_enemy_type)
    new_enemy_state = state.enemy_state.at[free_enemy_idx].set(jnp.array(1))


    new_enemy_x = jax.lax.cond(
        free_enemy_idx >= 0,
        lambda state: jax.lax.cond(
            new_enemy_type[free_enemy_idx] <= 1, # logic for boat or helicopter (spawn in water)
            lambda state: jax.lax.cond(
                state.river_inner_left[0] >= 0,
                lambda state: jax.lax.cond(
                    jax.random.bernoulli(x_key, 0.5),
                            lambda state: jax.random.randint(x_key, (), state.river_left[0] + 1, state.river_inner_left[0] - 8),
                            lambda state: jax.random.randint(x_key, (), state.river_inner_right[0] + 1, state.river_right[0] - 8),
                    operand=state
                ),
                lambda state: jax.random.randint(x_key, (), state.river_left[0] + 8, state.river_right[0] - 8),
                operand=state
            ),
            lambda state: jax.lax.cond( # logic for plane (select a screenside)
                jax.random.bernoulli(x_key, 0.5),
                lambda _: jnp.array(-10, dtype=jnp.int32),
                lambda _: jnp.array(SCREEN_WIDTH + 10, dtype=jnp.int32),
                operand=None
            ),
            operand=state
        ),
        lambda state: jnp.array(-1, dtype=jnp.int32),
        operand=state
    )
    new_enemy_y = jax.lax.cond(
        free_enemy_idx >= 0,
        lambda _: jnp.array(0, dtype=jnp.float32),
        lambda _: state.enemy_y[free_enemy_idx],
        operand=None
    )
    new_enemy_direction = jax.lax.cond(
        free_enemy_idx >= 0,
        lambda _: jax.random.randint(spawn_key, (), 0, 2), # 0 left, 1 right
        lambda _: state.enemy_direction[free_enemy_idx],
        operand=None
    )

    new_state = jax.lax.cond(
        free_enemy_idx >= 0,
        lambda new_state: new_state._replace(
            enemy_x=state.enemy_x.at[free_enemy_idx].set(new_enemy_x.astype(jnp.float32)),
            enemy_y=state.enemy_y.at[free_enemy_idx].set(new_enemy_y.astype(jnp.float32)),
            enemy_direction=state.enemy_direction.at[free_enemy_idx].set(new_enemy_direction),
            enemy_state=new_enemy_state,
            enemy_type=new_enemy_type
        ),
        lambda state: state,
        operand=state
    )
    return new_state

@jax.jit
def spawn_fuel(state: RiverraidState) -> RiverraidState:
    jax.debug.print("SPAWNING FUEL")
    return state

@jax.jit
def spawn_entities(state: RiverraidState) -> RiverraidState:
    key, subkey1, subkey2 = jax.random.split(state.master_key, 3)

    def spawn_entity(state: RiverraidState) -> RiverraidState:
        spawn_fuel_flag = jax.random.bernoulli(subkey2, 0.0) # TODO balance
        return jax.lax.cond(
            spawn_fuel_flag,
            lambda state: spawn_fuel(state),
            lambda state: spawn_enemies(state),
            operand=state
        )

    spawn_new_entity = jax.random.bernoulli(subkey1, 0.07) #TODO balance
    new_state = jax.lax.cond(
        jnp.logical_and(state.spawn_cooldown <= 0, spawn_new_entity),
        lambda state: spawn_entity(state),
        lambda state: state,
        operand=state
    )

    new_spawn_cooldown = jax.lax.cond(
        jnp.logical_and(state.spawn_cooldown <= 0, spawn_new_entity),
        lambda _: jnp.array(MINIMUM_SPAWN_COOLDOWN),
        lambda _: state.spawn_cooldown - 1,
        operand=None
    )

    return new_state._replace(master_key=key,
                              spawn_cooldown=new_spawn_cooldown)

def scroll_enemies(state: RiverraidState) -> RiverraidState:
    new_enemy_y = state.enemy_y + 1
    new_enemy_state = jnp.where(new_enemy_y > SCREEN_HEIGHT + 1, 0, state.enemy_state)
    new_enemy_x = jnp.where(new_enemy_y > SCREEN_HEIGHT + 1, -1, state.enemy_x)
    return state._replace(enemy_y=new_enemy_y,
                          enemy_state=new_enemy_state,
                          enemy_x=new_enemy_x)


def enemy_collision(state: RiverraidState) -> RiverraidState:
    def handle_bullet_collision(state: RiverraidState) -> RiverraidState:
        active_enemy_mask = state.enemy_state == 1

        x_collision_mask = (state.player_bullet_x < state.enemy_x + 8) & (state.player_bullet_x + 8 > state.enemy_x)
        y_collision_mask = (state.player_bullet_y < state.enemy_y + 8) & (state.player_bullet_y + 8 > state.enemy_y)

        collision_mask = active_enemy_mask & x_collision_mask & y_collision_mask
        collision_present = jnp.any(collision_mask)
        hit_index = jnp.argmax(collision_mask)

        new_enemy_state = jnp.where(
            collision_present,
            state.enemy_state.at[hit_index].set(0),
            state.enemy_state
        )

        new_bullet_x = jnp.where(collision_present, -1.0, state.player_bullet_x)
        new_bullet_y = jnp.where(collision_present, -1.0, state.player_bullet_y)

        return state._replace(
            enemy_state=new_enemy_state,
            player_bullet_x=new_bullet_x,
            player_bullet_y=new_bullet_y
        )

    # Bullet - Enemy Collision only when bullet present
    new_state = jax.lax.cond(
        state.player_bullet_y >= 0,
        lambda state: handle_bullet_collision(state),
        lambda state: state,
        state
    )

    # Player - Enemy Collision
    active_enemy_mask = new_state.enemy_state == 1

    x_collision = (new_state.player_x < new_state.enemy_x + 8) & (new_state.player_x + 8 > new_state.enemy_x)
    y_collision = (new_state.player_y < new_state.enemy_y + 8) & (new_state.player_y + 8 > new_state.enemy_y)
    collision_mask = active_enemy_mask & x_collision & y_collision


    collision_present = jnp.any(collision_mask)
    new_player_state = jnp.where(collision_present, 1, new_state.player_state)
    return new_state._replace(player_state=new_player_state)


@jax.jit
def update_enemy_movement_status(state: RiverraidState) -> RiverraidState:
    active_static_mask = (state.enemy_state == 1) & (state.enemy_direction <= 1)
    key, *subkeys = jax.random.split(state.master_key, MAX_ENEMIES + 1)
    subkeys = jnp.array(subkeys[:MAX_ENEMIES])

    def change_direction(i, enemy_direction):
        should_change = jax.lax.cond(
            active_static_mask[i],
            lambda _: jax.random.bernoulli(subkeys[i], 0.05),
            lambda _: False,
            operand=None
        )
        new_direction = jax.lax.cond(
            should_change,
            lambda _: jax.lax.cond(
                enemy_direction[i] == 0,
                lambda _: jnp.array(2),
                lambda _: jnp.array(3),
                operand=None
            ),
            lambda _: enemy_direction[i],
            operand=None
        )
        return enemy_direction.at[i].set(new_direction)

    new_enemy_direction = jax.lax.fori_loop(
        0, MAX_ENEMIES,
        lambda i, enemy_direction: change_direction(i, enemy_direction),
        state.enemy_direction
    )
    return state._replace(enemy_direction=new_enemy_direction, master_key=key)


def enemy_movement(state: RiverraidState) -> RiverraidState:
    new_enemy_x = state.enemy_x.copy()
    move_left_mask = (state.enemy_state == 1) & (state.enemy_direction == 2)
    move_right_mask = (state.enemy_state == 1) & (state.enemy_direction == 3)
    new_enemy_x = jnp.where(move_left_mask, new_enemy_x - 1, new_enemy_x)
    new_enemy_x = jnp.where(move_right_mask, new_enemy_x + 1, new_enemy_x)

    enemy_y = state.enemy_y.astype(jnp.int32)

    hit_left_bank = new_enemy_x <= state.river_left[enemy_y]
    hit_right_bank = new_enemy_x >= state.river_right[enemy_y] - 8

    hit_inner_left = (state.river_inner_left[enemy_y] >= 0) & (new_enemy_x <= state.river_inner_left[enemy_y])
    hit_inner_right = (state.river_inner_right[enemy_y] >= 0) & (new_enemy_x >= state.river_inner_right[enemy_y] - 8)

    change_direction_mask = hit_left_bank | hit_right_bank | hit_inner_left | hit_inner_right

    new_enemy_direction = jnp.where(
        change_direction_mask & (state.enemy_type != 2),
        jnp.where(state.enemy_direction == 2, 3, 2),
        state.enemy_direction
    )
    return state._replace(enemy_x=new_enemy_x, enemy_direction=new_enemy_direction)


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
                               island_transition_state=jnp.array(0),
                               segment_state=jnp.array(0),
                               segment_transition_state=jnp.array(0),
                               segment_straigt_counter=jnp.array(8),
                               dam_position= jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               player_x= jnp.array(SCREEN_WIDTH // 2 - 2, dtype=jnp.float32),
                               player_y=jnp.array(SCREEN_HEIGHT - 40),
                               player_velocity=jnp.array(0, dtype=jnp.float32),
                               player_direction=jnp.array(1),
                               player_state= jnp.array(0),
                               player_bullet_x= jnp.array(-1, dtype=jnp.float32),
                               player_bullet_y= jnp.array(-1, dtype=jnp.float32),
                               enemy_x=jnp.full((MAX_ENEMIES,), -1, dtype=jnp.float32),
                               enemy_y=jnp.full((MAX_ENEMIES,), SCREEN_HEIGHT + 1, dtype=jnp.float32),
                               enemy_state=jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                               enemy_type= jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                               enemy_direction= jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                               spawn_cooldown=jnp.array(50)
                               )
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def step(self, state: RiverraidState, action: Action) -> Tuple[RiverraidObservation, RiverraidState, RiverraidInfo]:
        def player_alive(state: RiverraidState) -> RiverraidState:
            new_state = roll_static_objects(state)
            new_state = update_river_banks(new_state)
            new_state = player_movement(new_state, action)
            new_state = player_shooting(new_state, action)
            new_state = spawn_entities(new_state)
            new_state = scroll_enemies(new_state)
            new_state = enemy_collision(new_state)
            new_state = update_enemy_movement_status(new_state)
            new_state = enemy_movement(new_state)
            return new_state

        def respawn(state: RiverraidState) -> RiverraidState:
            jax.debug.print("YOU DIED GIT GUD")
            #new_state = state._replace(player_state = jnp.array(0, dtype=state.player_state.dtype))
            return state

        jax.debug.print("new step \n")
        new_state = state._replace(turn_step=state.turn_step + 1)
        state = state._replace(player_state=jnp.array(0, dtype=state.player_state.dtype)) # immortal for testing
        new_state = jax.lax.cond(state.player_state == 0,
                                 lambda state: player_alive(state),
                                 lambda state: respawn(state),
                                 operand=new_state)



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


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/player.npy"),transpose=False)
    bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/bullet.npy"), transpose=False)
    enemy_boat = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/red_orange_enemy_1.npy"), transpose=False)
    enemy_helicopter = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/gray_enemy_1.npy"), transpose=False)
    enemy_airplane = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/purple_blue_enemy_1.npy"), transpose=False)

    SPRITE_PLAYER = jnp.expand_dims(player, axis = 0)
    BULLET = jnp.expand_dims(bullet, axis=0)
    ENEMY_BOAT = jnp.expand_dims(enemy_boat, axis=0)
    ENEMY_HELICOPTER = jnp.expand_dims(enemy_helicopter, axis=0)
    ENEMY_AIRPLANE = jnp.expand_dims(enemy_airplane, axis=0)
    return(
        SPRITE_PLAYER,
        BULLET,
        ENEMY_BOAT,
        ENEMY_HELICOPTER,
        ENEMY_AIRPLANE
    )

class RiverraidRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_PLAYER,
            self.BULLET,
            self.ENEMY_BOAT,
            self.ENEMY_HELICOPTER,
            self.ENEMY_AIRPLANE
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RiverraidState):
        green_banks = jnp.array([26, 132, 26], dtype=jnp.uint8)
        blue_river = jnp.array([42, 42, 189], dtype=jnp.uint8)
        dam_color = jnp.array([139, 69, 19], dtype=jnp.uint8)

        left_banks = state.river_left[:, None]
        right_banks = state.river_right[:, None]

        inner_left_banks = state.river_inner_left[:, None]
        inner_right_banks = state.river_inner_right[:, None]

        x_coords = jnp.arange(SCREEN_WIDTH)
        is_river =  (x_coords > left_banks) & (x_coords < right_banks) & jnp.logical_or(x_coords < inner_left_banks, x_coords > inner_right_banks)

        # The raster is  (HEIGHT, WIDTH, 3)
        raster = jnp.where(is_river[..., None], blue_river, green_banks)

        is_dam = (state.dam_position[:, None] == 1) & (x_coords > left_banks) & (x_coords < right_banks)
        raster = jnp.where(is_dam[..., None], dam_color, raster)

        # Player
        player_frame = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        px = jnp.round(state.player_x).astype(jnp.int32)
        py = jnp.round(state.player_y).astype(jnp.int32)
        raster = aj.render_at(raster, py, px, player_frame) # x and y swapped cuz its transposed later

        bullet_frame = aj.get_sprite_frame(self.BULLET, 0)
        bx = jnp.round(state.player_bullet_x).astype(jnp.int32)
        by = jnp.round(state.player_bullet_y).astype(jnp.int32)
        raster = aj.render_at(raster, by, bx, bullet_frame)

        def render_enemy_at_idx(raster, i):
            ex = jnp.round(state.enemy_x[i]).astype(jnp.int32)
            ey = jnp.round(state.enemy_y[i]).astype(jnp.int32)
            boat_frame = aj.get_sprite_frame(self.ENEMY_BOAT, 0)
            helicopter_frame = aj.get_sprite_frame(self.ENEMY_HELICOPTER, 0)
            airplane_frame = aj.get_sprite_frame(self.ENEMY_AIRPLANE, 0)
            frame_to_render = jax.lax.switch(
                state.enemy_type[i],
                [
                    lambda: boat_frame,
                    lambda: helicopter_frame,
                    lambda: airplane_frame,
                ]
            )
            return aj.render_at(raster, ey, ex, frame_to_render)

        def cond_fun(i):
            return state.enemy_state[i] > 0

        def body_fun(i, raster):
            raster = jax.lax.cond(
                cond_fun(i),
                lambda raster: render_enemy_at_idx(raster, i),
                lambda raster: raster,
                operand=raster
            )
            return raster

        raster = jax.lax.fori_loop(0, MAX_ENEMIES, body_fun, raster)

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

        action = get_action_from_keyboard(state)
        observation, state, reward, done, info = jitted_step(state, action)

        render_output = jitted_render(state)
        aj.update_pygame(screen, render_output, 1, SCREEN_WIDTH, SCREEN_HEIGHT)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()