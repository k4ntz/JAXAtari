import os
#from pty import spawn
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax

import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as aj


# -------- Game constants --------
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 200
DEFAULT_RIVER_WIDTH = 80
MIN_RIVER_WIDTH = 40
MAX_RIVER_WIDTH = 160
MAX_ENEMIES = 10
MINIMUM_SPAWN_COOLDOWN = 20
MAX_FUEL = 30
UI_HEIGHT = 35
SEGMENT_LENGTH = 400
DAM_OFFSET = 25
PLAYER_WIDTH = 7
PLAYER_HEIGHT = 14
DEATH_COOLDOWN = 50 # longer in real game
BUFFER = 50
ISLAND_SPAWN_PROB = 0.3
STRAIGHT_PROP = 0.6
EXPANSE_PROP = 0.2
SHRINK_PROP = 0.2
MIN_ALTERNATION_LENGTH = 3
MAX_ALTERNATION_LENGTH = 10
ENTITY_SPAWN_PROP = 0.2 # includes fuel, only checked if constraints met
FUEL_SPAWN_PROP = 0.1 # percentage of spawning entity being fuel
PLAYER_ACCELERATION = 0.02
PLAYER_MAX_SPEED = 4.0


class RiverraidState(NamedTuple):
    turn_step: chex.Array
    turn_step_linear: chex.Array
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
    housetree_position: chex.Array
    housetree_side: chex.Array
    housetree_direction: chex.Array
    housetree_cooldown: chex.Array

    player_x: chex.Array
    player_y: chex.Array
    player_velocity: chex.Array
    player_direction: chex.Array  # 0 left, 1 straight, 2 right
    player_state: chex.Array
    player_fuel: chex.Array
    player_score: chex.Array
    player_lives: chex.Array

    player_bullet_x: chex.Array
    player_bullet_y: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_state: chex.Array  # 0 empty/dead, 1 alive 2 dying first animation 3 dying second 4 dying final
    enemy_direction: chex.Array  # 0 left static, 1 right static, 2 left moving, 3 right moving
    fuel_x: chex.Array
    fuel_y: chex.Array
    fuel_state: chex.Array  # 0 empty, 1 present
    spawn_cooldown: chex.Array
    enemy_animation_cooldowns: chex.Array
    fuel_animation_cooldowns: chex.Array
    death_cooldown: chex.Array
    dam_explosion_cooldown: chex.Array



class RiverraidInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


class RiverraidObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_velocity: chex.Array
    player_fuel: chex.Array
    player_lives: chex.Array
    player_score: chex.Array
    river_left: chex.Array
    river_right: chex.Array
    river_inner_left: chex.Array
    river_inner_right: chex.Array
    dam_position: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_state: chex.Array
    enemy_direction: chex.Array
    fuel_x: chex.Array
    fuel_y: chex.Array
    fuel_state: chex.Array


# logic sperated into 3 branches: island, no_island, island_transition
# except for the transition phases, the states are managed at the top of the method
# islands are randomly spawned in no_island branch only when expanse is in progress (spawn trigger in expanse method)
# island_transition_state that manages the required straight segment at start and end
# islands can always shrink, if shrank below a threshold it is removed entirely
# the logic also tests wheter to despawn the island at any point (even when large)
@jax.jit
def generate_altering_river(state: RiverraidState) -> RiverraidState:
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

    # New state set here ONLY if NO alternation is ongoing
    # determine IF the river is to be altered, else use old state
    new_river_state = jax.lax.cond(
        jnp.logical_and(state.river_alternation_length <= 0, state.alternation_cooldown <= 0),
        lambda state: (jax.random.choice(
            alter_select_key, jnp.array([0, 1, 2]), p=jnp.array([STRAIGHT_PROP, EXPANSE_PROP, SHRINK_PROP])
        )),
        lambda state: state.river_state,
        operand=state
    )

    # alternation length is set here AND whenever an island spawns or despawns (due to the need for hard coding the segment)
    new_alternation_length = jax.lax.cond(state.river_alternation_length <= 0,
                                          lambda state: jax.random.randint(alter_length_key, (), MIN_ALTERNATION_LENGTH, MAX_ALTERNATION_LENGTH),
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
        return state._replace(river_left=new_river_left,
                              river_right=new_river_right,
                              river_inner_left=new_river_inner_left,
                              river_inner_right=new_river_inner_right)

    def no_island_branch(state: RiverraidState) -> RiverraidState:
        def expanse(state: RiverraidState) -> RiverraidState:
            new_river_left = state.river_left.at[0].set(state.river_left[1] - 3)
            new_river_right = state.river_right.at[0].set(state.river_right[1] + 3)

            key, subkey = jax.random.split(state.master_key)
            should_set_island = jnp.logical_and(jax.random.bernoulli(subkey, ISLAND_SPAWN_PROB),
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

            return state._replace(river_inner_left=new_river_inner_left,
                                  river_inner_right=new_river_inner_right,
                                  river_left=new_left,
                                  river_right=new_right,
                                  river_island_present=new_island_present
            )
        return lax.switch(
            state.river_state,
            [straight, expanse, shrink],
            state
        )


    def island_transition(state: RiverraidState) -> RiverraidState:
        def spawn_island(state: RiverraidState) -> RiverraidState:
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
                return state._replace(river_left=new_river_left,
                                      river_right=new_river_right,
                                      river_alternation_length= new_alternation_length,
                                      island_transition_state=new_island_transition_state)

            def initiate_split(state: RiverraidState) -> RiverraidState:
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
                new_river_left = state.river_left.at[0].set(state.river_left[1])
                new_river_right = state.river_right.at[0].set(state.river_right[1])

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

    # main handler for what branch to use
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


# segment transitions needs to ensure that no dam is present
# then shrink the river to dam size
# place a short straight segment before the dam
@jax.jit
def generate_segment_transition(state: RiverraidState) -> RiverraidState:
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
        # let the island removale be done by main river method by hard coding the river_island_present state to 3 (remove)
        return generate_altering_river(state._replace(river_island_present=new_island_present,
                                                      segment_transition_state=new_segment_transition_state))

    # let main method handle as explained above
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

    # place the dam and reset the states for next segment
    def dam_into_new_segment(state: RiverraidState) -> RiverraidState:
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
                              dam_explosion_cooldown=jnp.array(15)
                              )

    return jax.lax.switch(state.segment_transition_state,[first_call,
                                                                remove_island,
                                                                shrink_to_damsize,
                                                                straight_until_dam,
                                                                dam_into_new_segment],
                                                               operand=state)

# handles alternating segement - dam - straught segment - dam etc...
@jax.jit
def update_river_banks(state: RiverraidState) -> RiverraidState:
    new_segment_state = jax.lax.cond(
        (state.turn_step % SEGMENT_LENGTH) == 0,
        lambda state: state.segment_state + 1,
        lambda state: state.segment_state,
        operand=state
    )
    state = state._replace(segment_state=new_segment_state % 4)
    return jax.lax.switch(state.segment_state, [lambda state: generate_altering_river(state),
                                                lambda state: generate_segment_transition(state),
                                                lambda state: generate_straight_river(state),
                                                lambda state: generate_segment_transition(state)],
                                                operand=state)

# scroll dam and check collisions
@jax.jit
def handle_dam(state: RiverraidState) -> RiverraidState:
    new_dam_position = jnp.roll(state.dam_position, 1)
    new_dam_position = new_dam_position.at[-1].set(0)

    # Kollisionsprüfung Kugel <-> Damm
    bullet_y = jnp.round(state.player_bullet_y).astype(jnp.int32)
    bullet_x = jnp.round(state.player_bullet_x).astype(jnp.int32)

    dam_mask = new_dam_position == 1
    dam_exists = jnp.any(dam_mask)
    dam_top_y = jnp.argmax(dam_mask) - DAM_OFFSET
    bullet_above_dam = jnp.logical_and(
        dam_exists,
        jnp.logical_and(bullet_y < dam_top_y + DAM_OFFSET, bullet_x > 1)
    )

    river_left = state.river_left[bullet_y]
    river_right = state.river_right[bullet_y]

    new_bullet_x = jnp.where(bullet_above_dam, -1.0, state.player_bullet_x)
    new_bullet_y = jnp.where(bullet_above_dam, -1.0, state.player_bullet_y)

    player_above_dam = jnp.logical_and(dam_exists, state.player_y < dam_top_y + 25)
    new_player_state = jax.lax.cond(
        player_above_dam,
        lambda state: jnp.array(1),
        lambda state: state.player_state,
        operand=state
    )

    new_dam_position = jax.lax.cond(
        bullet_above_dam,
        lambda state: new_dam_position.at[dam_top_y + DAM_OFFSET].set(2),
        lambda state: new_dam_position,
        operand=state
    )
    return state._replace(
        dam_position=new_dam_position,
        player_bullet_x=new_bullet_x,
        player_bullet_y=new_bullet_y,
        player_state=new_player_state)

@jax.jit
def player_movement(state: RiverraidState, action: Action) -> RiverraidState:
    #input
    press_right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)
    press_left = (action == Action.LEFT) | (action == Action.LEFTFIRE)

    new_direction = 1 + press_right - press_left
    velocity_change = (press_right * PLAYER_ACCELERATION) - (press_left * PLAYER_ACCELERATION)
    current_velocity = jnp.where(
        (press_left == 0) & (press_right == 0),
        0.0,
        state.player_velocity + velocity_change
    )

    min_vel = jnp.where(press_right, 0.0, -PLAYER_MAX_SPEED)
    max_vel = jnp.where(press_left, 0.0, PLAYER_MAX_SPEED)
    new_velocity = jnp.clip(current_velocity, min_vel, max_vel)

    # move
    new_x = state.player_x + new_velocity

    # check collisions
    hitbox_left = new_x
    hitbox_right = new_x + PLAYER_WIDTH
    hitbox_top_y = jnp.clip(jnp.round(state.player_y).astype(jnp.int32), 0, SCREEN_HEIGHT - 1)
    hitbox_bottom_y = jnp.clip(jnp.round(state.player_y + PLAYER_HEIGHT).astype(jnp.int32), 0, SCREEN_HEIGHT - 1)

    collision_top_banks = (hitbox_left <= state.river_left[hitbox_top_y]) | \
                          (hitbox_right >= state.river_right[hitbox_top_y])
    collision_bottom_banks = (hitbox_left <= state.river_left[hitbox_bottom_y]) | \
                             (hitbox_right >= state.river_right[hitbox_bottom_y])

    island_present_top = state.river_inner_left[hitbox_top_y] >= 0
    collision_with_island_top = island_present_top & \
                                (hitbox_right >= state.river_inner_left[hitbox_top_y]) & \
                                (hitbox_left <= state.river_inner_right[hitbox_top_y])

    island_present_bottom = state.river_inner_left[hitbox_bottom_y] >= 0
    collision_with_island_bottom = island_present_bottom & \
                                   (hitbox_right >= state.river_inner_left[hitbox_bottom_y]) & \
                                   (hitbox_left <= state.river_inner_right[hitbox_bottom_y])

    any_collision = collision_top_banks | collision_bottom_banks | \
                    collision_with_island_top | collision_with_island_bottom

    # kill player if collision
    new_player_state = jnp.where(any_collision, 1, state.player_state)

    return state._replace(
        player_x=new_x,
        player_velocity=new_velocity,
        player_state=new_player_state,
        player_direction=new_direction
    )

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

@jax.jit
def player_shooting(state, action):
    shooting = action == Action.FIRE
    new_bullet_x, new_bullet_y = jax.lax.cond(
        jnp.logical_and(
            shooting,
            state.player_bullet_y < 0),
        lambda state: ((state.player_x + 3).astype(jnp.float32), (state.player_y - 0).astype(jnp.float32)),
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

    return state._replace(player_bullet_x=new_bullet_x,
                              player_bullet_y=new_bullet_y)


@jax.jit
def spawn_enemies(state):
    key, spawn_key, x_key = jax.random.split(state.master_key, 3)

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
    key, x_key = jax.random.split(state.master_key, 2)

    free_fuel_idx = jax.lax.cond(
        jnp.any(state.fuel_state == 0),
        lambda state: jnp.argmax(state.fuel_state == 0),
        lambda _: jnp.array(-1, dtype=jnp.int32),
        operand=state
    )

    new_fuel_state = state.fuel_state.at[free_fuel_idx].set(jnp.array(1))

    new_fuel_x = jax.lax.cond(
        free_fuel_idx >= 0,
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
        lambda state: jnp.array(-1, dtype=jnp.int32),
        operand=state
    )
    new_fuel_y = jax.lax.cond(
        free_fuel_idx >= 0,
        lambda _: jnp.array(0, dtype=jnp.float32),
        lambda _: state.enemy_y[free_fuel_idx],
        operand=None
    )

    return state._replace(fuel_state=new_fuel_state,
                            fuel_x=state.fuel_x.at[free_fuel_idx].set(new_fuel_x.astype(jnp.float32)),
                            fuel_y=state.fuel_y.at[free_fuel_idx].set(new_fuel_y.astype(jnp.float32)),
                            master_key=key)

@jax.jit
def spawn_entities(state: RiverraidState) -> RiverraidState:
    key, subkey1, subkey2 = jax.random.split(state.master_key, 3)

    def spawn_entity(state: RiverraidState) -> RiverraidState:
        spawn_fuel_flag = jax.random.bernoulli(subkey2, FUEL_SPAWN_PROP) # TODO balance
        return jax.lax.cond(
            spawn_fuel_flag,
            lambda state: spawn_fuel(state),
            lambda state: spawn_enemies(state),
            operand=state
        )
    # only spawn if no dam in the top 10 rows
    dam_at_top = jnp.any(state.dam_position[:50] >= 1)
    spawn_new_entity = jnp.logical_and(
        jax.random.bernoulli(subkey1, ENTITY_SPAWN_PROP), # TODO balance
        ~dam_at_top
    )

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

@jax.jit
def scroll_entities(state: RiverraidState) -> RiverraidState:
    new_enemy_y = state.enemy_y + 1
    new_enemy_state = jnp.where(new_enemy_y > SCREEN_HEIGHT + 1, 0, state.enemy_state)
    new_enemy_x = jnp.where(new_enemy_y > SCREEN_HEIGHT + 1, -1, state.enemy_x)

    new_fuel_y = state.fuel_y + 1
    new_fuel_state = jnp.where(new_fuel_y > SCREEN_HEIGHT + 1, 0, state.fuel_state)
    new_fuel_x = jnp.where(new_fuel_y > SCREEN_HEIGHT + 1, -1, state.fuel_x)

    return state._replace(
        enemy_y=new_enemy_y,
        enemy_state=new_enemy_state,
        enemy_x=new_enemy_x,
        fuel_y=new_fuel_y,
        fuel_state=new_fuel_state,
        fuel_x=new_fuel_x
    )

# river collision (direction turn) in enemy_movement
# handles bullet and player
@jax.jit
def enemy_collision(state: RiverraidState) -> RiverraidState:
    def handle_bullet_collision(state: RiverraidState) -> RiverraidState:
        enemy_hitboxes = jnp.array([
            12,  # boat
            8,  # helicopter
            6  # plane
        ])
        active_enemy_mask = state.enemy_state == 1

        hitboxes = enemy_hitboxes[state.enemy_type]

        x_collision_mask = ((state.player_bullet_x < state.enemy_x + hitboxes) & (state.player_bullet_x + hitboxes > state.enemy_x))
        y_collision_mask = ((state.player_bullet_y < state.enemy_y + 6) & (state.player_bullet_y + 1 > state.enemy_y))

        collision_mask = active_enemy_mask & x_collision_mask & y_collision_mask
        collision_present = jnp.any(collision_mask)
        hit_index = jnp.argmax(collision_mask)

        new_enemy_state = jnp.where(
            collision_present,
            state.enemy_state.at[hit_index].set(2),
            state.enemy_state
        )
        new_score = jnp.where(
            collision_present,
            state.player_score + jax.lax.switch(
                state.enemy_type[hit_index],
                [
                    lambda: 30,  # ship
                    lambda: 60,  # helicopter
                    lambda: 100  # plane
                ]
            ),
            state.player_score
        )

        new_bullet_x = jnp.where(collision_present, -1.0, state.player_bullet_x)
        new_bullet_y = jnp.where(collision_present, -1.0, state.player_bullet_y)

        return state._replace(
            enemy_state=new_enemy_state,
            player_bullet_x=new_bullet_x,
            player_bullet_y=new_bullet_y,
            player_score=new_score
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
def handle_animations(state: RiverraidState) -> RiverraidState:
    # enemies
    enemy_animation_mask = state.enemy_state > 1
    enemy_cooldown_positive_mask = state.enemy_animation_cooldowns > 0

    new_enemy_cooldowns = state.enemy_animation_cooldowns - 1

    enemy_state_change_mask = enemy_animation_mask & (new_enemy_cooldowns <= 0)

    new_enemy_state = jnp.where(enemy_state_change_mask & (state.enemy_state == 2), 3, state.enemy_state)
    new_enemy_state = jnp.where(enemy_state_change_mask & (state.enemy_state == 3), 4, new_enemy_state)
    new_enemy_state = jnp.where(enemy_state_change_mask & (state.enemy_state == 4), 0, new_enemy_state)

    new_enemy_cooldowns = jnp.where(enemy_state_change_mask, 10, new_enemy_cooldowns)

    # fuel
    fuel_animation_mask = state.fuel_state > 1
    new_fuel_cooldowns = state.fuel_animation_cooldowns - 1
    fuel_state_change_mask = fuel_animation_mask & (new_fuel_cooldowns <= 0)

    new_fuel_state = jnp.where(fuel_state_change_mask & (state.fuel_state == 2), 3, state.fuel_state)
    new_fuel_state = jnp.where(fuel_state_change_mask & (state.fuel_state == 3), 4, new_fuel_state)
    new_fuel_state = jnp.where(fuel_state_change_mask & (state.fuel_state == 4), 0, new_fuel_state)

    new_fuel_cooldowns = jnp.where(fuel_state_change_mask, 10, new_fuel_cooldowns)

    # dam
    new_dam_explosion_cooldown = jax.lax.cond(
        jnp.any(state.dam_position == 2) & (state.dam_explosion_cooldown > 0),
        lambda: state.dam_explosion_cooldown - 1,
        lambda: state.dam_explosion_cooldown
    )

    return state._replace(
        enemy_state=new_enemy_state,
        enemy_animation_cooldowns=new_enemy_cooldowns,
        fuel_state=new_fuel_state,
        fuel_animation_cooldowns=new_fuel_cooldowns,
        dam_explosion_cooldown=new_dam_explosion_cooldown
    )


# scrolling, player and bullet collision
@jax.jit
def handle_fuel(state: RiverraidState) -> RiverraidState:
    active_fuel_mask = state.fuel_state == 1

    # player collision
    x_collision_mask = (state.player_x < state.fuel_x + 8) & (state.player_x + 8 > state.fuel_x)
    y_collision_mask = (state.player_y < state.fuel_y + 8) & (state.player_y + 8 > state.fuel_y)
    player_collision_mask = active_fuel_mask & x_collision_mask & y_collision_mask
    player_collision_present = jnp.any(player_collision_mask)

    # bullet collision
    bullet_x_collision_mask = (state.player_bullet_x < state.fuel_x + 12) & (state.player_bullet_x + 0 > state.fuel_x)
    bullet_y_collision_mask = (state.player_bullet_y < state.fuel_y + 24) & (state.player_bullet_y > state.fuel_y)
    bullet_collision_mask = active_fuel_mask & bullet_x_collision_mask & bullet_y_collision_mask
    bullet_collision_present = jnp.any(bullet_collision_mask)
    bullet_hit_index = jnp.argmax(bullet_collision_mask)

    new_fuel_state = jnp.where(
        bullet_collision_present,
        state.fuel_state.at[bullet_hit_index].set(2),
        state.fuel_state
    )
    new_score = jnp.where(
        bullet_collision_present,
        state.player_score + 80,
        state.player_score
    )

    new_bullet_x = jnp.where(bullet_collision_present, -1.0, state.player_bullet_x)
    new_bullet_y = jnp.where(bullet_collision_present, -1.0, state.player_bullet_y)

    new_player_fuel = jax.lax.cond(
        player_collision_present,
        lambda state: jax.lax.cond(
            state.turn_step % 2 == 0,
            lambda state: jnp.clip(state.player_fuel + 1, 0, MAX_FUEL),
            lambda state: state.player_fuel,
            operand=state
        ),
        lambda state: jax.lax.cond(
            state.turn_step % 50 == 0,
            lambda state: jnp.clip(state.player_fuel - 1, 0, MAX_FUEL),
            lambda state: state.player_fuel,
            operand=state
        ),
        operand=state
    )

    return state._replace(
        fuel_state=new_fuel_state,
        player_bullet_x=new_bullet_x,
        player_bullet_y=new_bullet_y,
        player_fuel=new_player_fuel,
        player_score=new_score
    )

# enemies idle until randomly begin their movement
@jax.jit
def update_enemy_movement_status(state: RiverraidState) -> RiverraidState:
    active_static_mask = (state.enemy_state == 1) & (state.enemy_direction <= 1)
    key, subkey = jax.random.split(state.master_key, 2)

    should_change = jax.random.bernoulli(subkey, 0.01, shape=(MAX_ENEMIES,))
    should_change = should_change & active_static_mask

    # Bestimme neue Richtung basierend auf aktueller Richtung
    new_direction = jnp.where(
        state.enemy_direction == 0,
        jnp.array(2),  # left static -> left moving
        jnp.array(3)  # right static -> right moving
    )

    updated_direction = jnp.where(
        should_change,
        new_direction,
        state.enemy_direction
    )

    return state._replace(enemy_direction=updated_direction, master_key=key)

@jax.jit
def enemy_movement(state: RiverraidState) -> RiverraidState:
    new_enemy_x = state.enemy_x.copy()
    # Masks for enemies that are alive and set to a moving direction
    move_left_mask = (state.enemy_state == 1) & (state.enemy_direction == 2)
    move_right_mask = (state.enemy_state == 1) & (state.enemy_direction == 3)

    # Update positions based on direction
    new_enemy_x = jnp.where(move_left_mask, new_enemy_x - 0.5, new_enemy_x)
    new_enemy_x = jnp.where(move_right_mask, new_enemy_x + 0.5, new_enemy_x)

    enemy_y = state.enemy_y.astype(jnp.int32)
    enemy_width = 8

    # Collision with the outer river banks
    collides_with_outer_left = new_enemy_x <= state.river_left[enemy_y]
    collides_with_outer_right = new_enemy_x + enemy_width >= state.river_right[enemy_y]

    # Collision with the inner island shores
    island_present_at_y = state.river_inner_left[enemy_y] >= 0
    # A right-moving enemy hits the island's left shore
    collides_with_inner_left_shore = island_present_at_y & (new_enemy_x + enemy_width >= state.river_inner_left[enemy_y])
    # A left-moving enemy hits the island's right shore
    collides_with_inner_right_shore = island_present_at_y & (new_enemy_x <= state.river_inner_right[enemy_y])


    # For enemies moving left, only check for collisions on their left-hand side.
    left_side_collision = move_left_mask & (collides_with_outer_left | collides_with_inner_right_shore)

    # For enemies moving right, only check for collisions on their right-hand side.
    right_side_collision = move_right_mask & (collides_with_outer_right | collides_with_inner_left_shore)

    # Combine the masks to find all enemies that need to change direction
    change_direction_mask = left_side_collision | right_side_collision

    new_enemy_direction = jnp.where(
        change_direction_mask & (state.enemy_type != 2),  # Planes (type 2) fly off-screen and don't turn
        jnp.where(state.enemy_direction == 2, 3, 2),      # Flip direction from left (2) to right (3) or vice-versa
        state.enemy_direction
    )
    return state._replace(enemy_x=new_enemy_x, enemy_direction=new_enemy_direction)

@jax.jit
def handle_housetree(state: RiverraidState) -> RiverraidState:
    scrolled_housetree_position = jnp.roll(state.housetree_position, 1)
    scrolled_housetree_position = scrolled_housetree_position.at[0].set(-1)
    scrolled_houesetree_direction = jnp.roll(state.housetree_direction, 1)
    scrolled_houesetree_direction = scrolled_houesetree_direction.at[0].set(0)
    scrolled_housetree_side = jnp.roll(state.housetree_side, 1)
    scrolled_housetree_side = scrolled_housetree_side.at[0].set(0)
    new_state = state._replace(housetree_position=scrolled_housetree_position,
                               housetree_direction=scrolled_houesetree_direction,
                               housetree_side=scrolled_housetree_side,
                               housetree_cooldown=state.housetree_cooldown - 1)

    spawn_key, side_key, direction_key = jax.random.split(state.master_key, 3)

    def spawn_housetree(state: RiverraidState) -> RiverraidState:
        new_housetree_position = state.housetree_position.at[0].set(1)
        new_housetree_side = jax.lax.cond(
            jax.random.bernoulli(side_key, 0.5),
            lambda _: jnp.array(1),
            lambda _: jnp.array(2),
            operand=None
        )
        new_housetree_side = state.housetree_side.at[0].set(new_housetree_side)
        new_housetree_direction = jax.lax.cond(
            jax.random.bernoulli(direction_key, 0.5),
            lambda _: jnp.array(1),
            lambda _: jnp.array(2),
            operand=None
        )
        new_housetree_direction = state.housetree_direction.at[0].set(new_housetree_direction)
        return state._replace(housetree_position=new_housetree_position,
                              housetree_side=new_housetree_side,
                              housetree_direction=new_housetree_direction,
                              housetree_cooldown=jnp.array(50))

    spawn_new_housetree = jax.random.bernoulli(spawn_key, p=0.10)
    new_state = jax.lax.cond(
        jnp.logical_and(spawn_new_housetree, new_state.housetree_cooldown <= 0),
        lambda new_state: spawn_housetree(new_state),
        lambda new_state: new_state,
        operand=new_state
    )
    return new_state

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
        self.renderer = RiverraidRenderer()

    def render(self, state: RiverraidState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: RiverraidState) -> RiverraidObservation:
        return RiverraidObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            player_direction=state.player_direction,
            player_velocity=state.player_velocity,
            player_fuel=state.player_fuel,
            player_lives=state.player_lives,
            player_score=state.player_score,
            river_left=state.river_left,
            river_right=state.river_right,
            river_inner_left=state.river_inner_left,
            river_inner_right=state.river_inner_right,
            dam_position=state.dam_position,
            enemy_x=state.enemy_x,
            enemy_y=state.enemy_y,
            enemy_type=state.enemy_type,
            enemy_state=state.enemy_state,
            enemy_direction=state.enemy_direction,
            fuel_x=state.fuel_x,
            fuel_y=state.fuel_y,
            fuel_state=state.fuel_state,
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

    def reset(self, key=None) -> Tuple[RiverraidObservation, RiverraidState]:
        river_start_x = (SCREEN_WIDTH - DEFAULT_RIVER_WIDTH) // 2
        river_end_x = river_start_x + DEFAULT_RIVER_WIDTH
        initial_key = jax.random.PRNGKey(1)

        state = RiverraidState(turn_step=0,
                               turn_step_linear=0,
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
                               player_y=jnp.array(SCREEN_HEIGHT - 20 - UI_HEIGHT, dtype=jnp.float32),
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
                               fuel_x=jnp.full((MAX_ENEMIES,), -1, dtype=jnp.float32),
                               fuel_y=jnp.full((MAX_ENEMIES,), SCREEN_HEIGHT + 1, dtype=jnp.float32),
                               fuel_state=jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                               player_fuel=jnp.array(MAX_FUEL),
                               spawn_cooldown=jnp.array(50),
                               player_score=jnp.array(0),
                               player_lives=jnp.array(3),
                               housetree_position=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.float32),
                               housetree_side=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.float32),
                               housetree_direction=jnp.full((SCREEN_HEIGHT,), 0, dtype=jnp.float32),
                               housetree_cooldown=jnp.array(20),
                               enemy_animation_cooldowns=jnp.full(MAX_ENEMIES, 3),
                               fuel_animation_cooldowns=jnp.full(MAX_ENEMIES, 3),
                               death_cooldown=jnp.array(DEATH_COOLDOWN),
                               dam_explosion_cooldown=jnp.array(15)
                               )
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: RiverraidState, action: Action) -> Tuple[RiverraidObservation, RiverraidState, RiverraidInfo]:
        def player_alive(state: RiverraidState) -> RiverraidState:
            new_state = handle_dam(state)
            new_state = update_river_banks(new_state)
            new_state = player_movement(new_state, action)
            new_state = player_shooting(new_state, action)
            new_state = spawn_entities(new_state)
            new_state = scroll_entities(new_state)
            new_state = enemy_collision(new_state)
            new_state = handle_animations(new_state)
            new_state = update_enemy_movement_status(new_state)
            new_state = enemy_movement(new_state)
            new_state = handle_fuel(new_state)
            new_state = handle_housetree(new_state)
            #jax.debug.print("river island present : {}", new_state.river_island_present)
            #jax.debug.print("alternation cooldown : {}", new_state.alternation_cooldown)
            #jax.debug.print("alternation LENGTH : {}", new_state.river_alternation_length)
            return new_state

        def respawn(state: RiverraidState) -> RiverraidState:
            river_start_x = (SCREEN_WIDTH - DEFAULT_RIVER_WIDTH) // 2
            river_end_x = river_start_x + DEFAULT_RIVER_WIDTH
            initial_key = jax.random.PRNGKey(1)
            new_state = RiverraidState(turn_step= state.turn_step - (state.turn_step % SEGMENT_LENGTH + 50), # respawn at the start of the last segment + some offset
                                   turn_step_linear=state.turn_step_linear,
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
                                   dam_position=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                                   player_x=jnp.array(SCREEN_WIDTH // 2 - 2, dtype=jnp.float32),
                                   player_y=jnp.array(SCREEN_HEIGHT - 20 - UI_HEIGHT, dtype=jnp.float32),
                                   player_velocity=jnp.array(0, dtype=jnp.float32),
                                   player_direction=jnp.array(1),
                                   player_state=jnp.array(0),
                                   player_bullet_x=jnp.array(-1, dtype=jnp.float32),
                                   player_bullet_y=jnp.array(-1, dtype=jnp.float32),
                                   enemy_x=jnp.full((MAX_ENEMIES,), -1, dtype=jnp.float32),
                                   enemy_y=jnp.full((MAX_ENEMIES,), SCREEN_HEIGHT + 1, dtype=jnp.float32),
                                   enemy_state=jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   enemy_type=jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   enemy_direction=jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   fuel_x=jnp.full((MAX_ENEMIES,), -1, dtype=jnp.float32),
                                   fuel_y=jnp.full((MAX_ENEMIES,), SCREEN_HEIGHT + 1, dtype=jnp.float32),
                                   fuel_state=jnp.full((MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   player_fuel=jnp.array(MAX_FUEL),
                                   spawn_cooldown=jnp.array(50),
                                   player_score=state.player_score,
                                   player_lives= state.player_lives - 1,
                                   housetree_position=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.float32),
                                   housetree_side=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.float32),
                                   housetree_direction=jnp.full((SCREEN_HEIGHT,), -1, dtype=jnp.float32),
                                   housetree_cooldown=jnp.array(20),
                                   enemy_animation_cooldowns=jnp.full(MAX_ENEMIES, 3),
                                   fuel_animation_cooldowns=jnp.full(MAX_ENEMIES, 3),
                                   death_cooldown=jnp.array(DEATH_COOLDOWN),
                                   dam_explosion_cooldown=jnp.array(15)
                                   )
            return new_state

        def delay_respawn(state: RiverraidState) -> RiverraidState:
            new_death_cooldown = jnp.maximum(state.death_cooldown - 1, 0)
            return jax.lax.cond(
                new_death_cooldown <= 0,
                lambda state: respawn(state),
                lambda state: state._replace(death_cooldown=new_death_cooldown),
                operand=state
            )

        new_state = state._replace(turn_step=state.turn_step + 1,
                                   turn_step_linear=state.turn_step_linear + 1)

        new_player_state = jax.lax.cond(state.player_lives <= 0,
                                        lambda _: jnp.array(2),
                                        lambda _: new_state.player_state,
                                        operand=None
                                        )
        new_state = new_state._replace(player_state=new_player_state)  # game over

        new_state = jax.lax.cond(
            new_state.player_state == 0,
            lambda new_state: player_alive(new_state),
            lambda new_state: jax.lax.cond(
                new_state.player_state == 1,
                lambda new_state: delay_respawn(new_state),
                lambda new_state: new_state,
                operand=new_state
            ),
            operand = new_state
        )

        observation = self._get_observation(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state, all_rewards)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: RiverraidState) -> bool:
        return state.player_lives <= 0

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "player_x": spaces.Box(low=0 - BUFFER, high=SCREEN_WIDTH + BUFFER, shape=(), dtype=jnp.float32),
                "player_y": spaces.Box(low=0 - BUFFER, high=SCREEN_HEIGHT + BUFFER, shape=(), dtype=jnp.float32),
                "player_direction": spaces.Box(low=0 - BUFFER, high=2 + BUFFER, shape=(), dtype=jnp.float32),
                "player_velocity": spaces.Box(low=-3.0 - BUFFER, high=3.0 + BUFFER, shape=(), dtype=jnp.float32),
                "player_fuel": spaces.Box(low=0 - BUFFER, high=MAX_FUEL + BUFFER, shape=(), dtype=jnp.float32),
                "player_lives": spaces.Box(low=0 - BUFFER, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.float32),
                "player_score": spaces.Box(low=0 - BUFFER, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.float32),
                "river_left": spaces.Box(low=0 - BUFFER, high=SCREEN_WIDTH + BUFFER, shape=(SCREEN_HEIGHT,), dtype=jnp.float32),
                "river_right": spaces.Box(low=0 - BUFFER, high=SCREEN_WIDTH + BUFFER, shape=(SCREEN_HEIGHT,), dtype=jnp.float32),
                "river_inner_left": spaces.Box(low=-1 - BUFFER, high=SCREEN_WIDTH + BUFFER, shape=(SCREEN_HEIGHT,), dtype=jnp.float32),
                "river_inner_right": spaces.Box(low=-1 - BUFFER, high=SCREEN_WIDTH + BUFFER, shape=(SCREEN_HEIGHT,), dtype=jnp.float32),
                "dam_position": spaces.Box(low=-1 - BUFFER, high=2 + BUFFER, shape=(SCREEN_HEIGHT,), dtype=jnp.float32),
                "enemy_x": spaces.Box(low=-10 - BUFFER, high=SCREEN_WIDTH + 10 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "enemy_y": spaces.Box(low=0 - BUFFER, high=SCREEN_HEIGHT + 1 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "enemy_type": spaces.Box(low=0 - BUFFER, high=2 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "enemy_state": spaces.Box(low=0 - BUFFER, high=4 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "enemy_direction": spaces.Box(low=0 - BUFFER, high=3 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "fuel_x": spaces.Box(low=-1 - BUFFER, high=SCREEN_WIDTH + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "fuel_y": spaces.Box(low=0 - BUFFER, high=SCREEN_HEIGHT + 1 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
                "fuel_state": spaces.Box(low=0 - BUFFER, high=4 + BUFFER, shape=(MAX_ENEMIES,), dtype=jnp.float32),
            }
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: RiverraidState, state: RiverraidState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: RiverraidState, state: RiverraidState) -> chex.Array:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: RiverraidObservation) -> chex.Array:
        return jnp.concatenate(
            [
                obs.player_x.flatten(),
                obs.player_y.flatten(),
                obs.player_direction.flatten(),
                obs.player_velocity.flatten(),
                obs.player_fuel.flatten(),
                obs.player_lives.flatten(),
                obs.player_score.flatten(),
                obs.river_left.flatten(),
                obs.river_right.flatten(),
                obs.river_inner_left.flatten(),
                obs.river_inner_right.flatten(),
                obs.dam_position.flatten(),
                obs.enemy_x.flatten(),
                obs.enemy_y.flatten(),
                obs.enemy_type.flatten(),
                obs.enemy_state.flatten(),
                obs.enemy_direction.flatten(),
                obs.fuel_x.flatten(),
                obs.fuel_y.flatten(),
                obs.fuel_state.flatten(),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: RiverraidState, all_rewards: chex.Array = None) -> RiverraidInfo:
        """
        Creates the info object. The all_rewards argument is optional to handle
        the case where this is called during a reset() by the Gymnasium wrapper,
        where no rewards have been computed yet.
        """
        if all_rewards is None:
            # If no reward functions are defined, default to a single zero reward.
            if self.reward_funcs is None:
                all_rewards = jnp.zeros(1)
            # Otherwise, create a zero array matching the number of reward functions.
            else:
                all_rewards = jnp.zeros(len(self.reward_funcs))

        return RiverraidInfo(time=state.turn_step_linear, all_rewards=all_rewards)


def load_sprites():
    def normalize_frame(frame: jnp.ndarray, target_shape: Tuple[int, int, int]) -> jnp.ndarray:
        h, w, c = frame.shape
        th, tw, tc = target_shape
        assert c == tc, f"Channel mismatch: {c} vs {tc}"

        # Pad or crop vertically
        if h < th:
            top = (th - h) // 2
            bottom = th - h - top
            frame = jnp.pad(frame, ((top, bottom), (0, 0), (0, 0)), constant_values=0)
        elif h > th:
            crop = (h - th) // 2
            frame = frame[crop:crop + th, :, :]

        # Pad or crop horizontally
        if w < tw:
            left = (tw - w) // 2
            right = tw - w - left
            frame = jnp.pad(frame, ((0, 0), (left, right), (0, 0)), constant_values=0)
        elif w > tw:
            crop = (w - tw) // 2
            frame = frame[:, crop:crop + tw, :]

        return frame
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    player_center = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/plane_center.npy"),transpose=False)
    player_center = normalize_frame(player_center, (14, 7, 4))
    player_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/plane_left.npy"), transpose=False)
    player_left = normalize_frame(player_left, (14, 7, 4))
    player_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/plane_right.npy"), transpose=False)
    player_right = normalize_frame(player_right, (14, 7, 4))
    bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/bullet.npy"), transpose=False)
    enemy_boat = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/boat.npy"), transpose=False)
    enemy_boat = normalize_frame(enemy_boat, (24, 16, 4))
    enemy_boat = jnp.flip(enemy_boat, axis=1)
    enemy_helicopter = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/helicopter_1.npy"), transpose=False)
    enemy_helicopter = normalize_frame(enemy_helicopter, (24, 16, 4))
    enemy_helicopter_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/helicopter_2.npy"), transpose=False)
    enemy_helicopter_2 = normalize_frame(enemy_helicopter_2, (24, 16, 4))
    enemy_airplane = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/enemy_plane.npy"), transpose=False)
    enemy_airplane = normalize_frame(enemy_airplane, (24, 16, 4))
    enemy_airplane = jnp.flip(enemy_airplane, axis=1)
    fuel = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/fuel.npy"), transpose=False)
    fuel = normalize_frame(fuel, (24, 16, 4))
    fuel_display = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/fuel_display.npy"), transpose=False)
    fuel_indicator = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/fuel_indicator.npy"), transpose=False)
    house_tree = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/house_tree.npy"), transpose=False)
    full_dam = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/full_dam.npy"), transpose=False)
    outer_dam = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/outer_dam.npy"), transpose=False)
    street = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/street.npy"), transpose=False)
    activision = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/activision.npy"), transpose=False)
    explosion_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/explosion_1.npy"), transpose=False)
    explosion_1 = normalize_frame(explosion_1, (24, 16, 4))
    explosion_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/explosion_2.npy"), transpose=False)
    explosion_2 = normalize_frame(explosion_2, (24, 16, 4))
    player_explosion = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/explosion_2.npy"), transpose=False)
    player_explosion = normalize_frame(player_explosion, (14, 7, 4))
    dam_explosion_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/dam_explosion_1.npy"), transpose=False)
    dam_explosion_1 = normalize_frame(dam_explosion_1, (24, 16,4))
    dam_explosion_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/riverraid/dam_explosion_2.npy"), transpose=False)
    dam_explosion_2 = normalize_frame(dam_explosion_2, (24, 16,4))

    score_sprites = []
    for i in range(10):
        sprite_path = os.path.join(MODULE_DIR, f"sprites/riverraid/score_{i}.npy")
        score_sprite = aj.loadFrame(sprite_path, transpose=False)
        score_sprite = normalize_frame(score_sprite, (8, 6, 4))
        score_sprites.append(score_sprite)

    SPRITE_PLAYER = jnp.expand_dims(player_center, axis = 0)
    SPRITE_PLAYER_LEFT = jnp.expand_dims(player_left, axis=0)
    SPRITE_PLAYER_RIGHT = jnp.expand_dims(player_right, axis=0)
    BULLET = jnp.expand_dims(bullet, axis=0)
    ENEMY_BOAT = jnp.expand_dims(enemy_boat, axis=0)
    ENEMY_HELICOPTER = jnp.expand_dims(enemy_helicopter, axis=0)
    ENEMY_HELICOPTER_2 = jnp.expand_dims(enemy_helicopter_2, axis=0)
    ENEMY_AIRPLANE = jnp.expand_dims(enemy_airplane, axis=0)
    FUEL = jnp.expand_dims(fuel, axis=0)
    FUEL_DISPLAY = jnp.expand_dims(fuel_display, axis=0)
    FUEL_INDICATOR = jnp.expand_dims(fuel_indicator, axis=0)
    SPRITE_DIGIT = jnp.stack(score_sprites)
    HOUSE_TREE = jnp.expand_dims(house_tree, axis=0)
    FULL_DAM = jnp.expand_dims(full_dam, axis=0)
    OUTER_DAM = jnp.expand_dims(outer_dam, axis=0)
    STREET = jnp.expand_dims(street, axis=0)
    ACTIVISION = jnp.expand_dims(activision, axis=0)
    EXPLOSION_1 = jnp.expand_dims(explosion_1, axis=0)
    EXPLOSION_2 = jnp.expand_dims(explosion_2, axis=0)
    PLAYER_EXPLOSION = jnp.expand_dims(player_explosion, axis=0)
    DAM_EXPLOSION_1 = jnp.expand_dims(dam_explosion_1, axis=0)
    DAM_EXPLOSION_2 = jnp.expand_dims(dam_explosion_2, axis=0)

    return(
        SPRITE_PLAYER,
        SPRITE_PLAYER_LEFT,
        SPRITE_PLAYER_RIGHT,
        BULLET,
        ENEMY_BOAT,
        ENEMY_HELICOPTER,
        ENEMY_HELICOPTER_2,
        ENEMY_AIRPLANE,
        FUEL,
        FUEL_DISPLAY,
        FUEL_INDICATOR,
        SPRITE_DIGIT,
        HOUSE_TREE,
        FULL_DAM,
        OUTER_DAM,
        STREET,
        ACTIVISION,
        EXPLOSION_1,
        EXPLOSION_2,
        PLAYER_EXPLOSION,
        DAM_EXPLOSION_1,
        DAM_EXPLOSION_2
    )

class RiverraidRenderer(JAXGameRenderer):
    def __init__(self):
        (
            self.SPRITE_PLAYER,
            self.SPRITE_PLAYER_LEFT,
            self.SPRITE_PLAYER_RIGHT,
            self.BULLET,
            self.ENEMY_BOAT,
            self.ENEMY_HELICOPTER,
            self.ENEMY_HELICOPTER_2,
            self.ENEMY_AIRPLANE,
            self.FUEL,
            self.FUEL_DISPLAY,
            self.FUEL_INDICATOR,
            self.SPRITE_DIGIT,
            self.HOUSE_TREE,
            self.FULL_DAM,
            self.OUTER_DAM,
            self.STREET,
            self.ACTIVISION,
            self.EXPLOSION_1,
            self.EXPLOSION_2,
            self.PLAYER_EXPLOSION,
            self.DAM_EXPLOSION_1,
            self.DAM_EXPLOSION_2
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RiverraidState):
        green_banks = jnp.array([26, 132, 26], dtype=jnp.uint8)
        blue_river = jnp.array([42, 42, 189], dtype=jnp.uint8)
        red_river = jnp.array([189, 42, 42], dtype=jnp.uint8)
        ui_color = jnp.array([128, 128, 128], dtype=jnp.uint8)
        explosion_1 = aj.get_sprite_frame(self.EXPLOSION_1, 0)
        explosion_2 = aj.get_sprite_frame(self.EXPLOSION_2, 0)

        left_banks = state.river_left[:, None]
        right_banks = state.river_right[:, None]
        inner_left_banks = state.river_inner_left[:, None]
        inner_right_banks = state.river_inner_right[:, None]

        x_coords = jnp.arange(SCREEN_WIDTH)
        is_river = (x_coords > left_banks) & (x_coords < right_banks) & jnp.logical_or(
            x_coords < inner_left_banks, x_coords > inner_right_banks)

        # The raster is (HEIGHT, WIDTH, 3)
        river_color = jax.lax.cond(
            jnp.logical_and(
                state.dam_explosion_cooldown > 0,
                jnp.logical_and(
                    state.dam_explosion_cooldown < 15,
                    state.dam_explosion_cooldown % 2 == 0
                )
            ),
            lambda _: red_river,
            lambda _: blue_river,
            operand=None
        )
        raster = jnp.where(is_river[..., None], river_color, green_banks)

        # Render the dam sprite
        dam_y = jnp.argmax(state.dam_position >= 1).astype(jnp.int32) - DAM_OFFSET
        has_dam = jnp.max(state.dam_position) >= 1

        def render_dam(raster):
            dam_x = SCREEN_WIDTH // 2 - self.FULL_DAM.shape[2] // 2 + 5
            sprite_to_render = jax.lax.cond(
                state.dam_position[dam_y + DAM_OFFSET] == 1,
                lambda _: aj.get_sprite_frame(self.FULL_DAM, 0),
                lambda _: aj.get_sprite_frame(self.OUTER_DAM, 0),
                operand=None
            )
            raster = jax.lax.cond(
                has_dam,
                lambda raster: aj.render_at(raster, dam_x, dam_y, sprite_to_render),
                lambda raster: raster,
                operand=raster,
            )

            street = aj.get_sprite_frame(self.STREET, 0)
            raster = jax.lax.cond(
                has_dam,
                lambda raster: aj.render_at(raster, 0, dam_y, street), # fill in missing pixels
                lambda raster: raster,
                operand=raster,
            )

            explosion_to_render = jax.lax.cond(
                (state.dam_explosion_cooldown >= 5) & (state.dam_explosion_cooldown <= 10),
                lambda _: aj.get_sprite_frame(self.DAM_EXPLOSION_2, 0),
                lambda _: aj.get_sprite_frame(self.DAM_EXPLOSION_1, 0),
                operand=None
            )
            raster = jax.lax.cond(
                jnp.logical_and(state.dam_position[dam_y + DAM_OFFSET] == 2,
                                state.dam_explosion_cooldown > 0),
                lambda raster: aj.render_at(raster, SCREEN_WIDTH // 2 - 8, dam_y, explosion_to_render),
                lambda raster: raster,
                operand=raster
            )
            return raster

        raster = render_dam(raster)

        # Player
        player_frame = jax.lax.switch(
            state.player_direction,
            [
                lambda _: aj.get_sprite_frame(self.SPRITE_PLAYER_LEFT, 0),
                lambda _: aj.get_sprite_frame(self.SPRITE_PLAYER, 0),
                lambda _: aj.get_sprite_frame(self.SPRITE_PLAYER_RIGHT, 0)
            ],
            operand=None
        )
        px = jnp.round(state.player_x).astype(jnp.int32)
        py = jnp.round(state.player_y).astype(jnp.int32)
        player_frame = jax.lax.cond(
            state.player_state == 0, # alive
            lambda _: player_frame,
            lambda _: aj.get_sprite_frame(self.PLAYER_EXPLOSION, 0),
            operand=None
        )
        raster = aj.render_at(raster, px, py, player_frame)

        bullet_frame = aj.get_sprite_frame(self.BULLET, 0)
        bx = jnp.round(state.player_bullet_x).astype(jnp.int32)
        by = jnp.round(state.player_bullet_y).astype(jnp.int32)
        raster = aj.render_at(raster, bx, by, bullet_frame)

        def render_enemy_at_idx(raster, i):
            ex = jnp.round(state.enemy_x[i]).astype(jnp.int32)
            ey = jnp.round(state.enemy_y[i]).astype(jnp.int32)
            boat_frame = aj.get_sprite_frame(self.ENEMY_BOAT, 0)
            helicopter_frame = jax.lax.cond(
                state.turn_step % 2 == 0,
                lambda _: aj.get_sprite_frame(self.ENEMY_HELICOPTER, 0),
                lambda _: aj.get_sprite_frame(self.ENEMY_HELICOPTER_2, 0),
                operand=None
            )
            airplane_frame = aj.get_sprite_frame(self.ENEMY_AIRPLANE, 0)
            frame_to_render = jax.lax.switch(
                state.enemy_type[i],
                [
                    lambda: boat_frame,
                    lambda: helicopter_frame,
                    lambda: airplane_frame,
                ]
            )
            frame_to_render = jax.lax.cond(
                (state.enemy_direction[i] == 0) | (state.enemy_direction[i] == 2),
                lambda _: jnp.flip(frame_to_render, axis=1),
                lambda _: frame_to_render,
                operand=None
            )

            frame_to_render = jax.lax.cond(
                state.enemy_state[i] == 1,
                lambda _: frame_to_render,
                lambda _: jax.lax.cond(
                    state.enemy_state[i] == 2,
                    lambda _: explosion_1,
                    lambda _: jax.lax.cond(
                        state.enemy_state[i] == 3,
                        lambda _: explosion_2,
                        lambda _: explosion_1,
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )
            return aj.render_at(raster, ex, ey, frame_to_render)

        def render_alive_enemies(i, raster):
            raster = jax.lax.cond(
                state.enemy_state[i] > 0,
                lambda raster: render_enemy_at_idx(raster, i),
                lambda raster: raster,
                operand=raster
            )
            return raster

        raster = jax.lax.fori_loop(0, MAX_ENEMIES, render_alive_enemies, raster)

        def render_fuel_at_idx(raster, i):
            fx = jnp.round(state.fuel_x[i]).astype(jnp.int32)
            fy = jnp.round(state.fuel_y[i]).astype(jnp.int32)
            fuel_frame = aj.get_sprite_frame(self.FUEL, 0)

            # Determine which frame to render based on the fuel state
            frame_to_render = jax.lax.cond(
                state.fuel_state[i] == 1,
                lambda _: fuel_frame,
                lambda _: jax.lax.cond(
                    state.fuel_state[i] == 2,
                    lambda _: explosion_1,
                    lambda _: jax.lax.cond(
                        state.fuel_state[i] == 3,
                        lambda _: explosion_2,
                        lambda _: explosion_1,
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )

            return aj.render_at(raster, fx, fy, frame_to_render)

        def render_alive_fuel(i, raster):
            raster = jax.lax.cond(
                state.fuel_state[i] > 0,
                lambda raster: render_fuel_at_idx(raster, i),
                lambda raster: raster,
                operand=raster
            )
            return raster

        raster = jax.lax.fori_loop(0, MAX_ENEMIES, render_alive_fuel, raster)

        def render_housetree_at_idx(raster, i):
            hy = i
            hx = jax.lax.cond(
                state.housetree_side[i] == 1,
                lambda _: 5,
                lambda _: SCREEN_WIDTH - 20,
                operand=None
            )

            housetree_sprite = aj.get_sprite_frame(self.HOUSE_TREE, 0)
            housetree_sprite = jax.lax.cond(
                state.housetree_direction[i] == 2,
                lambda _: jnp.flip(housetree_sprite, axis=1),
                lambda _: housetree_sprite,
                operand=None
            )
            raster = jax.lax.cond(
                jnp.abs(hy - dam_y) > 20,
                lambda raster: aj.render_at(raster, hx, hy, housetree_sprite),
                lambda raster: raster,
                operand=raster
            )
            return raster

        def render_alive_housetree(i, raster):
            raster = jax.lax.cond(
                state.housetree_position[i] >= 0,
                lambda raster: render_housetree_at_idx(raster, i),
                lambda raster: raster,
                operand=raster
            )
            return raster

        raster = jax.lax.fori_loop(0, state.housetree_position.shape[0], render_alive_housetree, raster)

        # UI mask
        y_coords = jnp.arange(SCREEN_HEIGHT)
        ui_mask = y_coords >= (SCREEN_HEIGHT - UI_HEIGHT)
        raster = jnp.where(ui_mask[:, None, None], ui_color, raster)
        # black line between ui and fame
        raster = jax.lax.cond(
            SCREEN_HEIGHT - UI_HEIGHT > 0,
            lambda raster: raster.at[SCREEN_HEIGHT - UI_HEIGHT - 1].set(0),
            lambda raster: raster,
            operand=raster
        )

        fuel_frame = aj.get_sprite_frame(self.FUEL_DISPLAY, 0)
        fuel_display_x = SCREEN_WIDTH // 2 - fuel_frame.shape[1] // 2
        fuel_display_y = SCREEN_HEIGHT - UI_HEIGHT // 2 - fuel_frame.shape[0] // 2
        raster = aj.render_at(raster, fuel_display_x, fuel_display_y, fuel_frame)

        fuel_indicator_frame = aj.get_sprite_frame(self.FUEL_INDICATOR, 0)
        fuel_fill = state.player_fuel
        indicator_x = fuel_display_x + fuel_fill + 3
        raster = aj.render_at(raster, indicator_x, fuel_display_y + 4, fuel_indicator_frame)

        num_digits = jnp.maximum(1, jnp.ceil(jnp.log10(state.player_score + 1)).astype(jnp.int32))

        def get_digit(digit_place, score):
            digit = (score // jnp.power(10, digit_place)) % 10
            return digit.astype(jnp.int32)

        def score_loop_body(i, r_acc):
            # i is the position from the left (0 = leftmost digit, 1 = next, etc.).
            digit_place = num_digits - 1 - i
            digit_to_draw = get_digit(digit_place, state.player_score)

            x0 = jnp.int32(fuel_display_x + (i * 12))
            y0 = jnp.int32(fuel_display_y - 10)

            sprite_frame = aj.get_sprite_frame(self.SPRITE_DIGIT, digit_to_draw)
            return aj.render_at(r_acc, x0, y0, sprite_frame)
        raster = lax.fori_loop(0, num_digits, score_loop_body, raster)

        lives_frame = aj.get_sprite_frame(self.SPRITE_DIGIT, state.player_lives)
        lives_x = fuel_display_x - 8
        lives_y = fuel_display_y + fuel_frame.shape[0]
        raster = aj.render_at(raster, lives_x, lives_y, lives_frame)

        activision_sprite = aj.get_sprite_frame(self.ACTIVISION, 0)
        raster = aj.render_at(raster, SCREEN_WIDTH // 2 - activision_sprite.shape[1] // 2, SCREEN_HEIGHT - (activision_sprite.shape[0] + 1), activision_sprite)
        return raster


