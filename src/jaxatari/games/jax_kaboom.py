import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from pygame.examples.aliens import SCORE


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Kaboom.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {
            'name': 'background',
            'type': 'background',
            'file': 'background_border.npy'
        },
        {
            'name': 'bg_top',
            'type': 'single',
            'file': 'background_top.npy',
        },
        {
            'name': 'bg_bottom',
            'type': 'single',
            'file': 'background_bottom.npy',
        },
        {
            'name': 'mad_bomber',
            'type': 'single',
            'file': 'mad-bomber.npy',
        },
        {
            'name': 'bucket',
            'type': 'single',
            'file': 'bucket.npy',
        },
        {
            'name': 'bombs',
            'type': 'group',
            'files': (
                'bomb1.npy',
                'bomb2.npy',
            ),
        },
        {
            'name': 'bomb_fuse_states',
            'type': 'group',
            'files': [
                'fuse_anim1.npy',
                'fuse_anim2.npy',
                'fuse_anim3.npy',
            ],
        },
        {
            'name': 'bomb_explode_states',
            'type': 'group',
            'files': [
                'bomb_explode_anim1.npy',
                'bomb_explode_anim2.npy',
                'bomb_explode_anim3.npy',
            ],
        },
        {
            'name': 'bomb_bucket_explode_states',
            'type': 'group',
            'files': [
                'bomb_caught_anim1.npy',
                'bomb_caught_anim2.npy',
                'bomb_caught_anim3.npy',
            ],
        },
        {
            'name': 'scores',
            'type': 'group',
            'files': [
                'score0.npy',
                'score1.npy',
                'score2.npy',
                'score3.npy',
                'score4.npy',
                'score5.npy',
                'score6.npy',
                'score7.npy',
                'score8.npy',
                'score9.npy',
            ],
        },
    )


# Game contents
class KaboomConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    BUCKET_SPEED_X: int = 5  # in px
    BUCKET_X_OFFSET: int = 40  # in px
    BOMB_FUSE_STATES: int = 3  # bomb fuse animations count
    BOMB_EXPLODE_STATES: int = 11  # bomb animations count 4 + 4 + 3
    BOMB_BUCKET_EXPLODE_STATES: int = 12  # bomb animations count 4 + 4 + 4
    BOMB_SPAWN_HELP_VALUE_Y: int = 20
    BOMB_SIZE: tuple[int, int] = (5, 12)
    BACKGROUND_STATES: int = 32  # background flickers 16 times 2 frames each
    DEFAULT_STATE: int = -1
    MAXIMUM_SCORE: int = 999_999
    BOMBS_COUNT_GROUPS = jnp.array([10, 20, 30, 40, 50, 75, 100,150])  # bombs count
    MAD_BOMBER_SPEED_GROUPS = jnp.array([1, 2, 2, 3, 3, 4, 4, 4])  # in px
    MAD_BOMBER_RANDOMNESS_NUMBERS = jnp.array([1, 1, 2, 2, 3, 3, 4, 4])  # just numbers
    BOMB_SPEED_GROUPS = jnp.array([1, 1, 2, 2, 3, 3, 3, 4])  # in px
    BOMB_INTERVAL_PX_GROUPS = jnp.array([36, 18, 18, 10, 12, 6, 6, 3])  # in frames
    BACKGROUND_SIZE: tuple[int, int] = (160, 210)
    CUR_BACKGROUND_TOP: tuple[int, int] = (160, 210)
    BACKGROUND_TOP_SIZE: tuple[int, int] = (144, 40)
    BACKGROUND_TOP_POS: tuple[int, int] = (8, 7)
    SCORE_POS: tuple[int, int] = (92, 8)
    CUR_BACKGROUND_BOTTOM: tuple[int, int] = (144, 142)
    BACKGROUND_BOTTOM_SIZE: tuple[int, int] = (144, 142)
    BACKGROUND_BOTTOM_POS: tuple[int, int] = (8, 47)
    BUCKET_SIZE: tuple[int, int] = (14, 8)
    BOTTOM_EDGE_Y: int = 189
    MAD_BOMBER_SIZE: tuple[int, int] = (7, 30)
    MAD_BOMBER_POS_X: int = 22
    MAD_BOMBER_POS_Y: int = 20
    BUCKET_THREE_POS_X: int = 73
    BUCKET_THREE_POS_Y: int = 180
    BUCKET_TWO_POS_X: int = 73
    BUCKET_TWO_POS_Y: int = 164
    BUCKET_ONE_POS_X: int = 73
    BUCKET_ONE_POS_Y: int = 148
    TOPLEFT_ALLOWED_POS_X: int = 18
    TOPLEFT_ALLOWED_POS_Y: int = 128
    EXPIRES_IN_VALUES = jnp.array([i*12 for i in range(15)])
    ASSET_CONFIG: tuple = _get_default_asset_config()


# Agent's observation
class KaboomObservation(NamedTuple):
    mad_bomber_pos: chex.Array  # tuple[int, int]
    buckets_pos: chex.Array  # list[tuple[int, int]]
    bombs: chex.Array  # list[tuple[tuple[int, int], int, int, int, int, int]]  # EntityPosition, bomb_fuse_anim_state, bomb_type, explode_state, explode_bucket_state, on_bucket_index, is_active, explodes_in
    score: chex.Array
    lives: chex.Array


# Current game state
class KaboomState(NamedTuple):
    mad_bomber_pos_x: chex.Array
    mad_bomber_pos_y: chex.Array
    mad_bomber_going_left: chex.Array
    mad_bomber_motion_counter: chex.Array
    bombs_states: chex.Array
    buckets_pos: chex.Array
    buckets_moving_state: chex.Array
    buckets_jitter_state: chex.Array
    buckets_wereMovingRight: chex.Array
    score: chex.Array
    lives: chex.Array
    level: chex.Array
    frames_counter: chex.Array
    bombs_dropped: chex.Array
    bombs_falling_and_exploding: chex.Array
    bombs_exploding: chex.Array
    background_flickering: chex.Array
    background_state: chex.Array
    level_success: chex.Array
    level_finished: chex.Array
    key: chex.PRNGKey


class KaboomInfo(NamedTuple):
    score: chex.Array


class JaxKaboom(JaxEnvironment[KaboomState, KaboomObservation, KaboomInfo, KaboomConstants]):
    def __init__(self, consts: KaboomConstants = None):
        consts = consts or KaboomConstants()
        super().__init__(consts)
        self.renderer = KaboomRenderer(consts)
        self.action_set = [
            Action.RIGHT,
            Action.LEFT,
        ]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[KaboomObservation, KaboomState]:
        state_key, _step_key = jax.random.split(key)
        state = KaboomState(
            mad_bomber_pos_x=jnp.array(self.consts.MAD_BOMBER_POS_X, dtype=jnp.int32),
            mad_bomber_pos_y=jnp.array(self.consts.MAD_BOMBER_POS_Y, dtype=jnp.int32),
            mad_bomber_going_left=jnp.array(False, dtype=jnp.bool),
            mad_bomber_motion_counter=jnp.array(0),
            bombs_states=jnp.zeros((15, 9), dtype=jnp.int32), # x, y, bomb_fuse_anim_state, bomb_type, explode_state, explode_bucket_state, on_bucket_index, is_active, explodes_in
            buckets_pos=jnp.array([(self.consts.BUCKET_THREE_POS_X, self.consts.BUCKET_THREE_POS_Y, 1),
                                   (self.consts.BUCKET_TWO_POS_X, self.consts.BUCKET_TWO_POS_Y, 1),
                                   (self.consts.BUCKET_ONE_POS_X, self.consts.BUCKET_ONE_POS_Y, 1)], dtype=jnp.int32), # (x, y, is_active)
            buckets_jitter_state=jnp.array(self.consts.DEFAULT_STATE, dtype=jnp.int32),
            buckets_moving_state=jnp.array(self.consts.DEFAULT_STATE, dtype=jnp.int32),
            buckets_wereMovingRight=jnp.array(False),
            score=jnp.array(180000, dtype=jnp.int32),
            lives=jnp.array(3, dtype=jnp.int32),
            level=jnp.array(1, dtype=jnp.int32),
            bombs_dropped=jnp.array(0, dtype=jnp.int32),
            frames_counter=jnp.array(0, dtype=jnp.int32),
            bombs_falling_and_exploding=jnp.array(True, dtype=jnp.bool),
            bombs_exploding=jnp.array(False, dtype=jnp.bool),
            background_flickering=jnp.array(False, dtype=jnp.bool),
            background_state=jnp.array(self.consts.DEFAULT_STATE, dtype=jnp.int32),
            level_finished=jnp.array(False, dtype=jnp.bool),
            level_success=jnp.array(False, dtype=jnp.bool),
            key=state_key
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    def _get_observation(self, state: KaboomState):
        obs = KaboomObservation(
            mad_bomber_pos=jnp.array((state.mad_bomber_pos_x, state.mad_bomber_pos_y), dtype=jnp.int32),
            score=state.score,
            lives=state.lives,
            buckets_pos=state.buckets_pos,
            bombs=state.bombs_states,
        )
        return obs

    def update_bucket_step(self, state, action, bucket_pos, buckets_moving_state,
                           buckets_wereMovingRight, buckets_jitter_state,
                           frames_counter, key):

        # Skip if bombs exploding
        def skip_update(_):
            return bucket_pos, buckets_moving_state, buckets_wereMovingRight, buckets_jitter_state, frames_counter, key

        def do_update(operand):
            buckets_pos, buckets_moving_state, buckets_wereMovingRight, buckets_jitter_state, frames_counter, key = operand
            # Input + stickiness
            buckets_moving_state = jnp.where(
                (action == Action.LEFT) | (action == Action.RIGHT),
                jnp.minimum(5,
                            jnp.where(buckets_moving_state == self.consts.DEFAULT_STATE, 1, buckets_moving_state + 1)),
                jnp.maximum(0, buckets_moving_state - 1)
            )

            # Update movement direction
            buckets_wereMovingRight = jnp.where(action == Action.RIGHT, True,
                                                jnp.where(action == Action.LEFT, False, buckets_wereMovingRight))

            # Current speed
            cur_speed = jnp.where(
                buckets_moving_state == self.consts.DEFAULT_STATE,
                0,
                (self.consts.BUCKET_SPEED_X * (buckets_moving_state / 5)) * jnp.where(buckets_wereMovingRight, 1, -1)
            ).astype(int)

            # Tentative new_x
            new_x = jnp.where(
                buckets_wereMovingRight,
                jnp.minimum(self.consts.TOPLEFT_ALLOWED_POS_Y, buckets_pos[0, 0] + cur_speed),
                jnp.maximum(self.consts.TOPLEFT_ALLOWED_POS_X, buckets_pos[0, 0] + cur_speed)
            )

            # Bucket jittering
            key, subkey = jax.random.split(key)

            jitter_state_equals_default_state = (buckets_jitter_state == self.consts.DEFAULT_STATE)
            buckets_jitter_state = jnp.where(
                jitter_state_equals_default_state & (frames_counter % 100 == 0),
                jnp.where(jax.random.randint(subkey, (), 1, 4) == 3, 0, buckets_jitter_state),
                buckets_jitter_state
            )

            def apply_jitter(jitter_state):
                new_x_shift = jnp.where(jnp.isin(jitter_state, jnp.array([0, 1, 8, 9, 16, 17])), -1,
                                        jnp.where(jnp.isin(jitter_state, jnp.array([2, 3, 10, 11, 18, 19])), 1, 0))
                return new_x + new_x_shift

            frame_counter_even = (frames_counter % 2 == 0)
            new_x = jnp.where(
                ~jitter_state_equals_default_state & frame_counter_even,
                apply_jitter(buckets_jitter_state),
                new_x
            )
            buckets_jitter_state = jnp.where(~jitter_state_equals_default_state & frame_counter_even, buckets_jitter_state + 1, buckets_jitter_state)
            buckets_jitter_state = jnp.where(buckets_jitter_state == 29, self.consts.DEFAULT_STATE, buckets_jitter_state)

            buckets_pos = buckets_pos.at[:, 0].set(new_x)
            return buckets_pos, buckets_moving_state, buckets_wereMovingRight, buckets_jitter_state, frames_counter, key

        return jax.lax.cond(state.bombs_exploding, skip_update, do_update,
                            (bucket_pos, buckets_moving_state, buckets_wereMovingRight, buckets_jitter_state,
                             frames_counter, key))

    def get_group_index(self, level):
        return jax.lax.max(1, jax.lax.min(8, level)) - 1

    def update_bombs_step(self,
                          bombs,
                          buckets_pos,
                          mad_bomber_pos,
                          level,
                          bombs_dropped,
                          score,
                          level_finished,
                          level_success,
                          frames_counter,
                          bombs_falling_and_exploding,
                          bombs_should_explode,
                          key):
        X = 0
        Y = 1
        FUSE = 2
        TYPE = 3
        EXPLODE = 4
        BUCKET_EXPLODE = 5
        BUCKET_IDX = 6
        ACTIVE = 7
        EXPLODES_IN = 8
        def update_bombs_func_true(operand):
            bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key = operand

            # Dropping new bombs
            group_index = self.get_group_index(level)
            max_bombs = self.consts.BOMBS_COUNT_GROUPS[group_index]

            spawn_now = (
                    (bombs_dropped < max_bombs)
                    & (~level_finished)
                    & (frames_counter % self.consts.BOMB_INTERVAL_PX_GROUPS[group_index] == 0)
            )

            is_active = bombs[:, 7]
            free_mask = is_active == 0
            has_free_slot = jnp.any(free_mask)
            free_index = jnp.argmax(free_mask)  # safe because guarded
            def spawn_bomb(args):
                bombs, bombs_dropped, key, level_finished, level_success = args
                key, subkey = jax.random.split(key)

                new_bomb = jnp.array([
                    mad_bomber_pos[0] + 1,
                    mad_bomber_pos[1] + self.consts.BOMB_SPAWN_HELP_VALUE_Y,
                    0,
                    jax.random.randint(subkey, (), 0, 2),  # bomb_type
                    self.consts.DEFAULT_STATE,  # explode_state
                    self.consts.DEFAULT_STATE,  # explode_bucket_state
                    self.consts.DEFAULT_STATE,  # on_bucket_index
                    1,  # is_active
                    self.consts.DEFAULT_STATE  # expires_in
                ], dtype=jnp.int32)

                bombs = bombs.at[free_index].set(new_bomb)
                bombs_dropped = bombs_dropped + 1

                return bombs, bombs_dropped, key, level_finished, level_success

            def no_spawn(args):
                bombs, bombs_dropped, key, level_finished, level_success = args

                all_inactive = jnp.all(bombs[:, 7] == 0)
                level_finished = jnp.where(~(bombs_dropped < max_bombs) & all_inactive, True, level_finished)
                level_success = jnp.where(~(bombs_dropped < max_bombs) & all_inactive, True, level_success)

                return bombs, bombs_dropped, key, level_finished, level_success

            bombs, bombs_dropped, key, level_finished, level_success = jax.lax.cond(
                spawn_now & has_free_slot,
                spawn_bomb,
                no_spawn,
                (bombs, bombs_dropped, key, level_finished, level_success)
            )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # First for loop
            def update_bomb1(bomb):
                def active_case(operand):
                    bomb, scoreDelta = operand
                    x = bomb[X]
                    y = bomb[Y]
                    bucket_explode_state = bomb[BUCKET_EXPLODE]
                    bucket_idx = bomb[BUCKET_IDX]

                    # --------------------------------------------------
                    # 3. Bucket collision (only if not already exploding)
                    # --------------------------------------------------
                    def check_bucket():
                        def scan_bucket(j, res):
                            hit, idx = res
                            bucket = buckets_pos[j]
                            bucket_is_active = bucket[2] != 0

                            hit = (
                                    (y + self.consts.BOMB_SIZE[1] >= bucket[1])
                                    & (x + self.consts.BOMB_SIZE[0] >= bucket[0])
                                    & (x <= bucket[0] + self.consts.BUCKET_SIZE[0])
                            )
                            idx = jnp.where(bucket_is_active & hit & (idx == -1), j, idx)
                            return hit, idx

                        hit, idx = jax.lax.fori_loop(
                            0,
                            buckets_pos.shape[0],
                            scan_bucket,
                            (False, -1),
                        )

                        return (
                            jnp.where(hit, 0, bucket_explode_state),
                            jnp.where(hit, idx, bucket_idx),
                            jnp.where(hit, level, 0),
                        )
                    bucket_explode_state, bucket_idx, scoreDelta = jax.lax.cond(
                        jnp.logical_and(~bombs_should_explode, bucket_explode_state == self.consts.DEFAULT_STATE),
                        check_bucket,
                        lambda: (bucket_explode_state, bucket_idx, 0),
                    )
                    bucket_explode_state = jnp.where(
                        bucket_explode_state != self.consts.DEFAULT_STATE,
                        bucket_explode_state + 1,
                        bucket_explode_state,
                    )
                    bomb = bomb.at[BUCKET_EXPLODE].set(bucket_explode_state)\
                                .at[BUCKET_IDX].set(bucket_idx)
                    return bomb, scoreDelta

                return jax.lax.cond(
                    bomb[ACTIVE] == 1,
                    active_case,
                    lambda x: x,
                    operand=(bomb, 0),
                )
            bombs, all_score_delta = jax.vmap(update_bomb1)(bombs)
            score = score + jnp.sum(all_score_delta)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Second for loop
            hits_bottom = (~free_mask) & ((bombs[:, 1] + self.consts.BOMB_SIZE[1]) >= self.consts.BOTTOM_EDGE_Y)
            trigger_explosion = jnp.any(hits_bottom) & (~bombs_should_explode)
            bombs = bombs.at[:, EXPLODES_IN].set(
                jnp.where(trigger_explosion & (bombs[:, ACTIVE] == 1), self.consts.EXPIRES_IN_VALUES, bombs[:, EXPLODES_IN])
            )
            bombs_should_explode = bombs_should_explode | trigger_explosion
            level_finished = level_finished | trigger_explosion
            level_success = jnp.where(trigger_explosion, False, level_success)

            def update_bomb2(bomb):
                def active_case(operand):
                    bomb, bombs_should_explode, level_finished, level_success = operand
                    x = bomb[X]
                    y = bomb[Y]
                    fuse = bomb[FUSE]
                    explode_state = bomb[EXPLODE]
                    bucket_explode_state = bomb[BUCKET_EXPLODE]
                    bucket_idx = bomb[BUCKET_IDX]
                    explodes_in = bomb[EXPLODES_IN]

                    # --------------------------------------------------
                    # 2. Air explosion progression
                    # --------------------------------------------------
                    explode_state = jnp.where(
                        bombs_should_explode & (explodes_in == 0),
                        explode_state + 1,
                        explode_state,
                    )
                    explodes_in = jnp.where(explodes_in > 0, explodes_in - 1, explodes_in)

                    # --------------------------------------------------
                    # 5. Normal falling
                    # --------------------------------------------------
                    def fall():
                        key_, sub = jax.random.split(key)
                        new_fuse = jax.random.randint(
                            sub, (), 0, self.consts.BOMB_FUSE_STATES
                        )
                        return (
                            x,
                            y + self.consts.BOMB_SPEED_GROUPS[self.get_group_index(level)],
                            new_fuse,
                            self.consts.DEFAULT_STATE,
                            self.consts.DEFAULT_STATE,
                            bucket_idx,
                            key_,
                        )
                    x_, y_, fuse_, explode_state_, bucket_explode_state_, bucket_idx_, key_ = jax.lax.cond(
                        (~bombs_should_explode) & (bucket_explode_state == self.consts.DEFAULT_STATE),
                        fall,
                        lambda: (x, y, fuse, explode_state, bucket_explode_state, bucket_idx, key),
                    )

                    done = (
                            (explode_state_ >= self.consts.BOMB_EXPLODE_STATES)
                            | (bucket_explode_state_ >= self.consts.BOMB_BUCKET_EXPLODE_STATES)
                    )
                    bomb = jnp.array([
                        x_, y_, fuse_, bomb[TYPE],
                        explode_state_, bucket_explode_state_,
                        bucket_idx_, jnp.where(done, 0, bomb[ACTIVE]),
                        explodes_in
                    ])
                    return bomb, bombs_should_explode, level_finished, level_success

                return jax.lax.cond(
                    bomb[ACTIVE] == 1,
                    active_case,
                    lambda x: x,
                    operand=(bomb, bombs_should_explode, level_finished, level_success),
                )
            bombs, all_bombs_should_explode, all_level_finished, all_level_success= jax.vmap(update_bomb2)(bombs)
            bombs_should_explode = bombs_should_explode | jnp.any(all_bombs_should_explode)
            level_finished = level_finished | jnp.any(all_level_finished)
            level_success = level_success | jnp.any(all_level_success)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



            return bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return jax.lax.cond(bombs_falling_and_exploding, update_bombs_func_true, lambda x: x,
                            (bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key))

    def update_background_step(self, bombs, buckets_pos, bombs_should_explode, background_flickering, background_state, bombs_dropped, bombs_falling_and_exploding,
                               level_finished, level_success, level, lives):
        X = 0
        Y = 1
        FUSE = 2
        TYPE = 3
        EXPLODE = 4
        BUCKET_EXPLODE = 5
        BUCKET_IDX = 6
        ACTIVE = 7
        EXPLODES_IN = 8

        all_bombs_inactive = jnp.all(bombs[:, ACTIVE] == 0)
        start_flicker = bombs_should_explode & all_bombs_inactive

        bombs_should_explode = jnp.where(start_flicker, False, bombs_should_explode)
        background_flickering = jnp.where(start_flicker, True, background_flickering)
        bombs_dropped = jnp.where(start_flicker, 0, bombs_dropped)

        # ------------------------------------------------------------
        # Phase B1: level finished — failure path
        # ------------------------------------------------------------
        failure_step = (
                level_finished
                & (~level_success)
                & background_flickering
        )

        background_state = jnp.where(
            failure_step,
            background_state + 1,
            background_state,
        )

        animation_done = background_state >= self.consts.BACKGROUND_STATES

        def failure_reset():
            # deactivate the topmost active bucket
            active_mask = buckets_pos[:, 2] == 1
            remove_idx = jnp.argmax(active_mask)

            new_buckets = buckets_pos.at[remove_idx, 2].set(0)

            new_level = jnp.where(level > 1, level - 1, level)
            return (
                0,  # bombs_dropped
                True,  # bombs_falling_and_exploding
                False,  # level_finished
                False,  # level_success
                False,  # background_flickering
                self.consts.DEFAULT_STATE,  # background_state
                lives - 1,
                new_level,
                new_buckets,
            )

        (
            bombs_dropped,
            bombs_falling_and_exploding,
            level_finished,
            level_success,
            background_flickering,
            background_state,
            lives,
            level,
            buckets_pos,
        ) = jax.lax.cond(
            failure_step & animation_done,
            failure_reset,
            lambda: (
                bombs_dropped,
                bombs_falling_and_exploding,
                level_finished,
                level_success,
                background_flickering,
                background_state,
                lives,
                level,
                buckets_pos,
            ),
        )

        # ------------------------------------------------------------
        # Phase B2: level finished — success path
        # ------------------------------------------------------------
        success_reset = (
                level_finished
                & level_success
                & all_bombs_inactive
        )

        bombs_dropped = jnp.where(success_reset, 0, bombs_dropped)
        bombs_falling_and_exploding = jnp.where(success_reset, True, bombs_falling_and_exploding)
        level_finished = jnp.where(success_reset, False, level_finished)
        level_success = jnp.where(success_reset, False, level_success)
        level = jnp.where(success_reset, level + 1, level)

        # ------------------------------------------------------------
        # Return updated state
        # ------------------------------------------------------------
        return bombs_should_explode, background_flickering, background_state, bombs_dropped, bombs_falling_and_exploding, level_finished, level_success, level, lives, buckets_pos

    def update_mad_bomber_step(self, mad_bomber_motion_counter, mad_bomber_pos_x, mad_bomber_going_left, level,
                               level_finished, bombs_dropped, key):
        group = self.get_group_index(level)

        can_move = (
                (bombs_dropped < self.consts.BOMBS_COUNT_GROUPS[group]) &
                (~level_finished)
        )

        speed = self.consts.MAD_BOMBER_SPEED_GROUPS[group]
        randomness = self.consts.MAD_BOMBER_RANDOMNESS_NUMBERS[group]

        # Horizontal movement (only if counter != 0)
        move_delta = jnp.where(
            mad_bomber_going_left,
            -speed,
            speed,
        )

        mad_bomber_pos_x = jnp.where(
            can_move & (mad_bomber_motion_counter != 0),
            mad_bomber_pos_x + move_delta,
            mad_bomber_pos_x,
        )

        # Random direction reset every 18 frames
        reset_motion = can_move & ((mad_bomber_motion_counter // 18) == 1)

        key, subkey = jax.random.split(key)
        random_dir = jax.random.randint(subkey, (), 0, 2).astype(bool)

        mad_bomber_motion_counter = jnp.where(
            reset_motion,
            0,
            mad_bomber_motion_counter,
        )

        mad_bomber_going_left = jnp.where(
            reset_motion,
            random_dir,
            mad_bomber_going_left,
        )

        # Boundary checks
        mad_bomber_going_left = jnp.where(
            mad_bomber_pos_x >= self.consts.TOPLEFT_ALLOWED_POS_Y,
            True,
            mad_bomber_going_left,
        )

        mad_bomber_going_left = jnp.where(
            mad_bomber_pos_x <= self.consts.TOPLEFT_ALLOWED_POS_X,
            False,
            mad_bomber_going_left,
        )

        # Advance motion counter
        mad_bomber_motion_counter = jnp.where(
            can_move,
            mad_bomber_motion_counter + randomness,
            mad_bomber_motion_counter,
        )

        return (
            mad_bomber_motion_counter,
            mad_bomber_pos_x,
            mad_bomber_going_left,
            key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: KaboomState, action: chex.Array) -> Tuple[
        KaboomObservation, KaboomState, float, bool, KaboomInfo]:
        # if state.lives <= 0:
        #     return self._get_observation(state), state, 0, True, self._get_info(state)
        #
        # if state.score >= self.consts.MAXIMUM_SCORE:
        #     return self._get_observation(state), state, 0, True, self._get_info(state)

        buckets_pos, buckets_moving_state, buckets_wereMovingRight, buckets_jitter_state, frames_counter, key \
            = self.update_bucket_step(state, action, state.buckets_pos, state.buckets_moving_state,
                                      state.buckets_wereMovingRight, state.buckets_jitter_state, state.frames_counter,
                                      state.key)

        bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key \
            = self.update_bombs_step(state.bombs_states, buckets_pos,
                                     (state.mad_bomber_pos_x, state.mad_bomber_pos_y), state.level, state.bombs_dropped,
                                     state.score, state.level_finished, state.level_success, frames_counter,
                                     state.bombs_falling_and_exploding, state.bombs_exploding, key)

        bombs_should_explode, background_flickering, background_state, bombs_dropped, bombs_falling_and_exploding, level_finished, level_success, level, lives, buckets_pos \
            = self.update_background_step(bombs, buckets_pos, bombs_should_explode, state.background_flickering, state.background_state, bombs_dropped, bombs_falling_and_exploding,
                               level_finished, level_success, level, state.lives)

        mad_bomber_motion_counter, mad_bomber_pos_x, mad_bomber_going_left, key \
            = self.update_mad_bomber_step(state.mad_bomber_motion_counter, mad_bomber_pos[0],
                                          state.mad_bomber_going_left, level, level_finished, bombs_dropped, key)

        frames_counter += 1
        new_state = KaboomState(
            mad_bomber_pos_x=mad_bomber_pos_x,
            mad_bomber_pos_y=mad_bomber_pos[1],
            mad_bomber_going_left=mad_bomber_going_left,
            mad_bomber_motion_counter=mad_bomber_motion_counter,
            bombs_states=bombs,
            buckets_pos=buckets_pos,
            buckets_jitter_state=buckets_jitter_state,
            buckets_moving_state=buckets_moving_state,
            buckets_wereMovingRight=buckets_wereMovingRight,
            score=score,
            lives=lives,
            level=level,
            bombs_dropped=bombs_dropped,
            frames_counter=frames_counter,
            bombs_falling_and_exploding=bombs_falling_and_exploding,
            bombs_exploding=bombs_should_explode,
            background_flickering=background_flickering,
            background_state=background_state,
            level_finished=level_finished,
            level_success=level_success,
            key=key
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: KaboomState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: KaboomState) -> KaboomInfo:
        return KaboomInfo(score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KaboomState, state: KaboomState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: KaboomState) -> bool:
        return state.lives <= 0

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: KaboomObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""
        return jnp.concatenate([
            obs.mad_bomber_pos.flatten(),
            obs.buckets_pos.flatten(),
            obs.bombs.flatten(),
            obs.score.flatten(),
            obs.lives.flatten()
        ])


class KaboomRenderer(JAXGameRenderer):
    def __init__(self, consts: KaboomConstants = None):
        super().__init__()
        self.consts = consts or KaboomConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/kaboom"

        # 2. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        # Pre-stack all related sprites for easy indexing in the render loop
        self.BACKGROUNDS = [self.SHAPE_MASKS['bg_top'], self.SHAPE_MASKS['bg_bottom']]
        self.BOMBS = self.SHAPE_MASKS['bombs']
        self.BOMB_FUSE_STATES = self.SHAPE_MASKS['bomb_fuse_states']
        self.BOMB_EXPLODE_STATES = self.SHAPE_MASKS['bomb_explode_states']
        self.BOMB_BUCKET_EXPLODE_STATES = self.SHAPE_MASKS['bomb_bucket_explode_states']
        self.SCORES = self.SHAPE_MASKS['scores']

        # Create randomized backgrounds
        self.COLORED_BACKGROUNDS_STACKED = self._create_random_backgrounds()

    def _create_random_backgrounds(self, n_variants: int = 10, key=None):
        """
        Create a stack of random backgrounds based on the 2D background masks.

        Each pixel is replaced by a random value (like fill() in PyGame).
        Keeps everything 2D to match self.BACKGROUNDS.

        Returns:
            jnp.Array of shape (n_variants, 3, H, W)
            - 3 = (top, bottom, border)
        """
        backgrounds = self.BACKGROUNDS  # tuple: (top, bottom, border)
        if key is None:
            key = jax.random.PRNGKey(0)

        H, W = backgrounds[0].shape  # assume all backgrounds have same shape
        stacked_backgrounds = []

        for i in range(n_variants):
            key, subkey_top, subkey_bottom, subkey_border = jax.random.split(key, 4)

            bg_top = jax.random.randint(subkey_top, shape=(H, W), minval=0, maxval=256)
            bg_bottom = jax.random.randint(subkey_bottom, shape=(H, W), minval=0, maxval=256)
            bg_border = jax.random.randint(subkey_border, shape=(H, W), minval=0, maxval=256)

            # Stack top, bottom, border into one array per variant
            bg_variant = jnp.stack([bg_top, bg_bottom, bg_border], axis=0)  # (3, H, W)
            stacked_backgrounds.append(bg_variant)

        return jnp.stack(stacked_backgrounds, axis=0)  # (n_variants, 3, H, W)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KaboomState) -> chex.Array:
        # # ------------------------------------------------------------
        # # 1. Hintergrund auswählen (inkl. Flicker)
        # # ------------------------------------------------------------
        # bg_variant_idx = state.background_state % self.COLORED_BACKGROUNDS_STACKED.shape[0]
        # bg_stack = self.COLORED_BACKGROUNDS_STACKED[bg_variant_idx]  # (3, H, W)
        #
        # # Falls kein Flicker → Originalbackground
        # bg_stack = jax.lax.cond(
        #     state.background_flickering,
        #     lambda _: bg_stack,
        #     lambda _: jnp.stack(self.BACKGROUNDS, axis=0),
        #     operand=None
        # )

        # Background_border
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Background_top
        raster = self.jr.render_at(
            raster,
            x=self.consts.BACKGROUND_TOP_POS[0],
            y=self.consts.BACKGROUND_TOP_POS[1],
            sprite_mask=self.BACKGROUNDS[0],  # top
        )

        # Background_bottom
        raster = self.jr.render_at(
            raster,
            x=self.consts.BACKGROUND_BOTTOM_POS[0],
            y=self.consts.BACKGROUND_BOTTOM_POS[1],
            sprite_mask=self.BACKGROUNDS[1],  # top
        )

        # Score
        score = state.score

        score_digit0 = self.jr.int_to_digits(score // 1000000 % 10, max_digits=1)
        raster = jax.lax.cond(jnp.all(score_digit0 != 0),
                              lambda _: self.jr.render_label(
                                  raster,
                                  self.consts.SCORE_POS[0] - 42,
                                  self.consts.SCORE_POS[1],
                                  score_digit0,
                                  self.SCORES,
                                  spacing=-7,
                                  max_digits=1
                              ),
                              lambda _: raster, operand=None)

        score_digit1 = self.jr.int_to_digits(score // 100000 % 10, max_digits=1)
        raster = jax.lax.cond(jnp.all(score_digit1 != 0) | jnp.all(score_digit0 != 0),
                              lambda _: self.jr.render_label(
                                  raster,
                                  self.consts.SCORE_POS[0] - 35,
                                  self.consts.SCORE_POS[1],
                                  score_digit1,
                                  self.SCORES,
                                  spacing=-7,
                                  max_digits=1
                              ),
                              lambda _: raster, operand=None)

        score_digit2 = self.jr.int_to_digits(score // 10000 % 10, max_digits=1)
        raster = jax.lax.cond(jnp.all(score_digit2 != 0) | jnp.all(score_digit1 != 0) | jnp.all(score_digit0 != 0),
                              lambda _: self.jr.render_label(
                                  raster,
                                  self.consts.SCORE_POS[0] - 28,
                                  self.consts.SCORE_POS[1],
                                  score_digit2,
                                  self.SCORES,
                                  spacing=-7,
                                  max_digits=1
                              ),
                              lambda _: raster, operand=None)

        score_digit3 = self.jr.int_to_digits(score // 1000 % 10, max_digits=1)
        raster = jax.lax.cond(jnp.all(score_digit3 != 0) | jnp.all(score_digit2 != 0) | jnp.all(score_digit1 != 0) | jnp.all(score_digit0 != 0),
                              lambda _: self.jr.render_label(
                                  raster,
                                  self.consts.SCORE_POS[0] - 21,
                                  self.consts.SCORE_POS[1],
                                  score_digit3,
                                  self.SCORES,
                                  spacing=-7,
                                  max_digits=1
                              ),
                              lambda _: raster, operand=None)

        score_digit4 = self.jr.int_to_digits(score // 100 % 10, max_digits=1)
        raster = jax.lax.cond(jnp.all(score_digit4 != 0) | jnp.all(score_digit3 != 0) | jnp.all(score_digit2 != 0) | jnp.all(score_digit1 != 0) | jnp.all(score_digit0 != 0),
                              lambda _: self.jr.render_label(
                                  raster,
                                  self.consts.SCORE_POS[0] - 14,
                                  self.consts.SCORE_POS[1],
                                  score_digit4,
                                  self.SCORES,
                                  spacing=-7,
                                  max_digits=1
                              ),
                              lambda _: raster, operand=None)

        score_digit5 = self.jr.int_to_digits(score // 10 % 10, max_digits=1)
        raster = jax.lax.cond(jnp.all(score_digit5 != 0) | jnp.all(score_digit4 != 0) | jnp.all(score_digit3 != 0) | jnp.all(score_digit2 != 0) | jnp.all(score_digit1 != 0) | jnp.all(score_digit0 != 0),
                     lambda _: self.jr.render_label(
                        raster,
                        self.consts.SCORE_POS[0] - 7,
                        self.consts.SCORE_POS[1],
                        score_digit5,
                        self.SCORES,
                        spacing=-7,
                        max_digits=1
                    ),
                     lambda _: raster, operand=None)

        score_digit6 = self.jr.int_to_digits(score % 10, max_digits=1)
        raster = self.jr.render_label(
                        raster,
                        self.consts.SCORE_POS[0],
                        self.consts.SCORE_POS[1],
                        score_digit6,
                        self.SCORES,
                        spacing=-7,
                        max_digits=1
                    )


        # Mad bomber
        raster = self.jr.render_at(
            raster,
            state.mad_bomber_pos_x.astype(int),
            state.mad_bomber_pos_y.astype(int),
            self.SHAPE_MASKS["mad_bomber"],
        )


        # Bombs
        def draw_bomb(i, r):
            bomb = state.bombs_states[i]
            is_active = bomb[7] != 0
            #jax.lax.cond(is_active, lambda: jax.debug.print("{}", bomb), lambda: None)

            def draw_active(r2):
                x = bomb[0].astype(int)
                y = bomb[1].astype(int)

                # --- Explosion in Luft ---
                def draw_air_explosion(r3):
                    idx = (bomb[4] > 3).astype(jnp.int32) + (bomb[4] > 7).astype(jnp.int32)
                    return self.jr.render_at(r3, x, y, self.BOMB_EXPLODE_STATES[idx])

                # --- Explosion im Eimer ---
                def draw_bucket_explosion(r3):
                    idx = (bomb[5] > 3).astype(jnp.int32) + (bomb[5] > 7).astype(jnp.int32)
                    bucket = state.buckets_pos[bomb[6]]
                    by = bucket[1] - self.BOMB_BUCKET_EXPLODE_STATES[idx].shape[0]
                    return self.jr.render_at(r3, bucket[0], by, self.BOMB_BUCKET_EXPLODE_STATES[idx])

                # --- Normale Bombe ---
                def draw_normal(r3):
                    r3 = self.jr.render_at(r3, x, y, self.BOMBS[bomb[3]])
                    fuse_x = x + 2 * ((bomb[3] + 1) % 2)
                    fuse_y = y - self.BOMB_FUSE_STATES[bomb[3]].shape[0]
                    fuse_x = jnp.where(bomb[2] == 1, fuse_x + 1, fuse_x)
                    fuse_y = jnp.where(bomb[2] == 1, fuse_y + 2, fuse_y)
                    return self.jr.render_at(r3, fuse_x, fuse_y, self.BOMB_FUSE_STATES[bomb[2]])

                return jax.lax.cond(
                    bomb[4] != self.consts.DEFAULT_STATE,
                    draw_air_explosion,
                    lambda r4: jax.lax.cond(
                        bomb[5] != self.consts.DEFAULT_STATE,
                        draw_bucket_explosion,
                        draw_normal,
                        r4,
                    ),
                    r2,
                )

            return jax.lax.cond(is_active, draw_active, lambda r2: r2, r)

        raster = jax.lax.fori_loop(
            0,
            state.bombs_states.shape[0],
            draw_bomb,
            raster,
        )

        # Buckets
        def draw_bucket(i, r):
            bucket = state.buckets_pos[i]
            jax.debug.print("{}", bucket)
            return jax.lax.cond(
                bucket[2] != 0,
                lambda r2: self.jr.render_at(r2, bucket[0], bucket[1], self.SHAPE_MASKS["bucket"]),
                lambda r2: r2,
                r,
            )

        raster = jax.lax.fori_loop(
            0,
            state.buckets_pos.shape[0],
            draw_bucket,
            raster,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)