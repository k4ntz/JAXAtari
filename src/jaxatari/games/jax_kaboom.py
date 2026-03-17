import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.modification import AutoDerivedConstants
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


def get_default_asset_config() -> tuple:
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
            'file': 'mad_bomber.npy',
        },
        {
            'name': 'bucket',
            'type': 'single',
            'file': 'bucket.npy',
        },
        {
            'name': 'bombs',
            'type': 'group',
            'files': [
                'bomb1.npy',
                'bomb2.npy',
            ],
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


class KaboomConstants(AutoDerivedConstants):
    EXPIRES_IN_VALUES: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168])
    )
    ASSET_CONFIG: tuple = struct.field(
        pytree_node=False,
        default_factory=get_default_asset_config
    )
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    BUCKET_SPEED_X: int = struct.field(pytree_node=False, default=5)  # in px
    BUCKET_X_OFFSET: int = struct.field(pytree_node=False, default=40)  # in px
    BOMB_FUSE_STATES: int = struct.field(pytree_node=False, default=3)  # bomb fuse animations count
    BOMB_EXPLODE_STATES: int = struct.field(pytree_node=False, default=11)  # bomb animations count 4 + 4 + 3
    BOMB_BUCKET_EXPLODE_STATES: int = struct.field(pytree_node=False, default=12)  # bomb animations count 4 + 4 + 4
    BOMB_SPAWN_HELP_VALUE_Y: int = struct.field(pytree_node=False, default=20)
    BOMB_SIZE: tuple[int, int] = struct.field(pytree_node=False, default=(5, 12))
    BACKGROUND_STATES: int = struct.field(pytree_node=False, default=32)  # background flickers 16 times 2 frames each
    DEFAULT_STATE: int = struct.field(pytree_node=False, default=-1)
    MAX_SCORE: int = struct.field(pytree_node=False, default=999_999)
    BOMBS_COUNT_GROUPS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([10, 20, 30, 40, 50, 75, 100, 150])
    )  # bombs count
    MAD_BOMBER_SPEED_GROUPS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1, 2, 2, 3, 3, 4, 4, 4])
    )  # in px
    MAD_BOMBER_RANDOMNESS_NUMBERS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1, 1, 2, 2, 3, 3, 4, 4])
    )  # just numbers
    BOMB_SPEED_GROUPS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1, 1, 2, 2, 3, 3, 3, 4])
    )  # in px
    BOMB_INTERVAL_PX_GROUPS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([36, 18, 18, 10, 12, 6, 6, 3])
    )  # in frames
    BACKGROUND_SIZE: tuple[int, int] = struct.field(pytree_node=False, default=(160, 210))
    CUR_BACKGROUND_TOP: tuple[int, int] = struct.field(pytree_node=False, default=(160, 210))
    BACKGROUND_TOP_SIZE: tuple[int, int] = struct.field(pytree_node=False, default=(144, 40))
    BACKGROUND_TOP_POS: tuple[int, int] = struct.field(pytree_node=False, default=(8, 7))
    SCORE_POS: tuple[int, int] = struct.field(pytree_node=False, default=(92, 8))
    CUR_BACKGROUND_BOTTOM: tuple[int, int] = struct.field(pytree_node=False, default=(144, 142))
    BACKGROUND_BOTTOM_SIZE: tuple[int, int] = struct.field(pytree_node=False, default=(144, 142))
    BACKGROUND_BOTTOM_POS: tuple[int, int] = struct.field(pytree_node=False, default=(8, 47))
    BUCKET_SIZE: tuple[int, int] = struct.field(pytree_node=False, default=(14, 8))
    BOTTOM_EDGE_Y: int = struct.field(pytree_node=False, default=189)
    MAD_BOMBER_SIZE: tuple[int, int] = struct.field(pytree_node=False, default=(7, 30))
    MAD_BOMBER_POS_X: int = struct.field(pytree_node=False, default=22)
    MAD_BOMBER_POS_Y: int = struct.field(pytree_node=False, default=20)
    BUCKET_THREE_POS_X: int = struct.field(pytree_node=False, default=73)
    BUCKET_THREE_POS_Y: int = struct.field(pytree_node=False, default=180)
    BUCKET_TWO_POS_X: int = struct.field(pytree_node=False, default=73)
    BUCKET_TWO_POS_Y: int = struct.field(pytree_node=False, default=164)
    BUCKET_ONE_POS_X: int = struct.field(pytree_node=False, default=73)
    BUCKET_ONE_POS_Y: int = struct.field(pytree_node=False, default=148)
    BUCKET_MIN_ALLOWED_POS_X: int = struct.field(pytree_node=False, default=18)
    BUCKET_MAX_ALLOWED_POS_X: int = struct.field(pytree_node=False, default=128)


@struct.dataclass
class KaboomObservation:
    mad_bomber_pos: ObjectObservation  # tuple[int, int]
    buckets_pos: ObjectObservation
    bombs: ObjectObservation
    score: ObjectObservation
    lives: ObjectObservation


@struct.dataclass
class KaboomState:
    mad_bomber_pos_x: chex.Array
    mad_bomber_pos_y: chex.Array
    mad_bomber_going_left: chex.Array
    mad_bomber_motion_counter: chex.Array
    bombs_states: chex.Array
    buckets_pos: chex.Array
    buckets_moving_state: chex.Array
    buckets_jitter_state: chex.Array
    buckets_were_moving_right: chex.Array
    score: chex.Array
    lives: chex.Array
    level: chex.Array
    frames_counter: chex.Array
    bombs_dropped: chex.Array
    bombs_falling_and_exploding: chex.Array
    bombs_exploding: chex.Array
    background_flickering: chex.Array
    background_state: chex.Array
    waiting_for_restart: chex.Array
    level_success: chex.Array
    level_finished: chex.Array
    key: chex.PRNGKey


@struct.dataclass
class KaboomInfo:
    score: chex.Array


class JaxKaboom(JaxEnvironment[KaboomState, KaboomObservation, KaboomInfo, KaboomConstants]):
    def __init__(self, consts: KaboomConstants = None):
        super().__init__(consts)
        self.consts = consts or KaboomConstants()
        self.renderer = KaboomRenderer(self.consts)
        self.action_set = [
            Action.RIGHT,
            Action.LEFT,
            Action.FIRE,
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
            buckets_were_moving_right=jnp.array(False),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(3, dtype=jnp.int32),
            level=jnp.array(1, dtype=jnp.int32),
            bombs_dropped=jnp.array(0, dtype=jnp.int32),
            frames_counter=jnp.array(0, dtype=jnp.int32),
            bombs_falling_and_exploding=jnp.array(True, dtype=jnp.bool),
            bombs_exploding=jnp.array(False, dtype=jnp.bool),
            background_flickering=jnp.array(False, dtype=jnp.bool),
            background_state=jnp.array(self.consts.DEFAULT_STATE, dtype=jnp.int32),
            waiting_for_restart=jnp.array(False, dtype=jnp.bool),
            level_finished=jnp.array(False, dtype=jnp.bool),
            level_success=jnp.array(False, dtype=jnp.bool),
            key=state_key
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: KaboomState):
        obs = KaboomObservation(
            mad_bomber_pos=jnp.array((state.mad_bomber_pos_x, state.mad_bomber_pos_y), dtype=jnp.int32),
            score=state.score,
            lives=state.lives,
            buckets_pos=state.buckets_pos,
            bombs=state.bombs_states,
        )
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def update_bucket_step(self, state, action, bucket_pos, buckets_moving_state,
                           buckets_were_moving_right, buckets_jitter_state,
                           frames_counter, key):

        def do_update(operand):
            buckets_pos, buckets_moving_state, buckets_were_moving_right, buckets_jitter_state, frames_counter, key = operand
            # Input + stickiness
            buckets_moving_state = jnp.where(
                (action == Action.LEFT) | (action == Action.RIGHT),
                jnp.minimum(5,
                            jnp.where(buckets_moving_state == self.consts.DEFAULT_STATE, 1, buckets_moving_state + 1)),
                jnp.maximum(0, buckets_moving_state - 1)
            )

            # Update movement direction
            buckets_were_moving_right = jnp.where(action == Action.RIGHT, True,
                                                jnp.where(action == Action.LEFT, False, buckets_were_moving_right))

            # Current speed
            cur_speed = jnp.where(
                buckets_moving_state == self.consts.DEFAULT_STATE,
                0,
                (self.consts.BUCKET_SPEED_X * (buckets_moving_state / 5)) * jnp.where(buckets_were_moving_right, 1, -1)
            ).astype(int)

            # Tentative new_x
            new_x = jnp.where(
                buckets_were_moving_right,
                jnp.minimum(self.consts.BUCKET_MAX_ALLOWED_POS_X, buckets_pos[0, 0] + cur_speed),
                jnp.maximum(self.consts.BUCKET_MIN_ALLOWED_POS_X, buckets_pos[0, 0] + cur_speed)
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
            return buckets_pos, buckets_moving_state, buckets_were_moving_right, buckets_jitter_state, frames_counter, key

        return jax.lax.cond(state.bombs_exploding, lambda x: x, do_update,
                            (bucket_pos, buckets_moving_state, buckets_were_moving_right, buckets_jitter_state,
                             frames_counter, key))

    @partial(jax.jit, static_argnums=(0,))
    def get_group_index(self, level):
        return jax.lax.max(1, jax.lax.min(8, level)) - 1

    def bomb_move_horizontally(self, bomb_pos_x, min_allowed_pos_x, max_allowed_pos_x, subkey) -> chex.Array:
        return bomb_pos_x

    def bomb_move_vertically(self, bomb_pos_y,  subkey) -> chex.Array:
        return bomb_pos_y

    @partial(jax.jit, static_argnums=(0,))
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

        def maybe_spawn_bomb(bombs, bombs_dropped, level_finished, level_success, key):
            group_index = self.get_group_index(level)
            max_bombs = self.consts.BOMBS_COUNT_GROUPS[group_index]
            spawn_interval = self.consts.BOMB_INTERVAL_PX_GROUPS[group_index]

            spawn_now = (
                (bombs_dropped < max_bombs)
                & (~level_finished)
                & (frames_counter % spawn_interval == 0)
            )

            free_mask = bombs[:, ACTIVE] == 0
            has_free_slot = jnp.any(free_mask)
            free_index = jnp.argmax(free_mask)

            def do_spawn(args):
                bombs_, bombs_dropped_, key_, level_finished_, level_success_ = args
                key_, subkey = jax.random.split(key_)
                new_bomb = jnp.array([
                    mad_bomber_pos[0] + 1,
                    mad_bomber_pos[1] + self.consts.BOMB_SPAWN_HELP_VALUE_Y,
                    0,
                    jax.random.randint(subkey, (), 0, 2),
                    self.consts.DEFAULT_STATE,
                    self.consts.DEFAULT_STATE,
                    self.consts.DEFAULT_STATE,
                    1,
                    self.consts.DEFAULT_STATE,
                ], dtype=jnp.int32)

                bombs_ = bombs_.at[free_index].set(new_bomb)
                return bombs_, bombs_dropped_ + 1, key_, level_finished_, level_success_

            def skip_spawn(args):
                bombs_, bombs_dropped_, key_, level_finished_, level_success_ = args
                all_inactive = jnp.all(bombs_[:, ACTIVE] == 0)
                level_complete = (~(bombs_dropped_ < max_bombs)) & all_inactive
                level_finished_ = jnp.where(level_complete, True, level_finished_)
                level_success_ = jnp.where(level_complete, True, level_success_)
                return bombs_, bombs_dropped_, key_, level_finished_, level_success_

            return jax.lax.cond(
                spawn_now & has_free_slot,
                do_spawn,
                skip_spawn,
                (bombs, bombs_dropped, key, level_finished, level_success),
            )

        def update_bucket_collision(bomb):
            def active_case(operand):
                bomb_, score_delta = operand
                x = bomb_[X]
                y = bomb_[Y]
                bucket_explode_state = bomb_[BUCKET_EXPLODE]
                bucket_idx = bomb_[BUCKET_IDX]

                def check_bucket_collision():
                    def scan_bucket(j, res):
                        hit_any, first_idx = res
                        bucket = buckets_pos[j]
                        bucket_is_active = bucket[2] != 0
                        hit = (
                            (y + self.consts.BOMB_SIZE[1] >= bucket[1])
                            & (x + self.consts.BOMB_SIZE[0] >= bucket[0])
                            & (x <= bucket[0] + self.consts.BUCKET_SIZE[0])
                        )
                        should_capture = bucket_is_active & hit & (first_idx == -1)
                        hit_any = hit_any | should_capture
                        first_idx = jnp.where(should_capture, j, first_idx)
                        return hit_any, first_idx

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

                bucket_explode_state, bucket_idx, score_delta = jax.lax.cond(
                    (~bombs_should_explode) & (bucket_explode_state == self.consts.DEFAULT_STATE),
                    check_bucket_collision,
                    lambda: (bucket_explode_state, bucket_idx, 0),
                )

                bucket_explode_state = jnp.where(
                    bucket_explode_state != self.consts.DEFAULT_STATE,
                    bucket_explode_state + 1,
                    bucket_explode_state,
                )

                bomb_ = (
                    bomb_.at[BUCKET_EXPLODE].set(bucket_explode_state)
                    .at[BUCKET_IDX].set(bucket_idx)
                )
                return bomb_, score_delta

            return jax.lax.cond(
                bomb[ACTIVE] == 1,
                active_case,
                lambda x: x,
                operand=(bomb, 0),
            )

        def trigger_ground_explosion(bombs, bombs_should_explode, level_finished, level_success):
            hits_bottom = (
                (bombs[:, ACTIVE] == 1)
                & ((bombs[:, Y] + self.consts.BOMB_SIZE[1]) >= self.consts.BOTTOM_EDGE_Y)
            )
            trigger_explosion = jnp.any(hits_bottom) & (~bombs_should_explode)

            def assign_ordered_explosion_delays(bombs_):
                active_mask = bombs_[:, ACTIVE] == 1
                sortable_y = jnp.where(active_mask, bombs_[:, Y], jnp.array(-1, dtype=jnp.int32))
                sorted_indices = jnp.argsort(-sortable_y)
                active_sorted = active_mask[sorted_indices]
                active_ranks = jnp.cumsum(active_sorted.astype(jnp.int32)) - 1
                sorted_delays = jnp.where(
                    active_sorted,
                    self.consts.EXPIRES_IN_VALUES[active_ranks],
                    self.consts.DEFAULT_STATE,
                )
                delays = jnp.full((bombs_.shape[0],), self.consts.DEFAULT_STATE, dtype=jnp.int32)
                delays = delays.at[sorted_indices].set(sorted_delays)
                return bombs_.at[:, EXPLODES_IN].set(
                    jnp.where(active_mask, delays, bombs_[:, EXPLODES_IN])
                )

            bombs = jax.lax.cond(
                trigger_explosion,
                assign_ordered_explosion_delays,
                lambda b: b,
                bombs,
            )
            bombs_should_explode = bombs_should_explode | trigger_explosion
            level_finished = level_finished | trigger_explosion
            level_success = jnp.where(trigger_explosion, False, level_success)
            return bombs, bombs_should_explode, level_finished, level_success

        def advance_bomb_state(bomb, bomb_key):
            def active_case(operand):
                bomb_, bombs_should_explode_, level_finished_, level_success_, bomb_key_ = operand
                x = bomb_[X]
                y = bomb_[Y]
                fuse = bomb_[FUSE]
                explode_state = bomb_[EXPLODE]
                bucket_explode_state = bomb_[BUCKET_EXPLODE]
                bucket_idx = bomb_[BUCKET_IDX]
                explodes_in = bomb_[EXPLODES_IN]

                explode_state = jnp.where(
                    bombs_should_explode_ & (explodes_in == 0),
                    explode_state + 1,
                    explode_state,
                )
                explodes_in = jnp.where(explodes_in > 0, explodes_in - 1, explodes_in)

                def fall():
                    fuse_key, move_key = jax.random.split(bomb_key_)
                    new_fuse = jax.random.randint(
                        fuse_key, shape=(), minval=0, maxval=self.consts.BOMB_FUSE_STATES, dtype=jnp.int32
                    )
                    new_x = self.bomb_move_horizontally(
                        x,
                        self.consts.BUCKET_MIN_ALLOWED_POS_X,
                        self.consts.BUCKET_MAX_ALLOWED_POS_X,
                        move_key,
                    )

                    move_key, fall_key = jax.random.split(bomb_key_)
                    new_y = self.bomb_move_vertically(y, fall_key)

                    return (
                        new_x,
                        new_y + self.consts.BOMB_SPEED_GROUPS[self.get_group_index(level)],
                        new_fuse,
                        self.consts.DEFAULT_STATE,
                        self.consts.DEFAULT_STATE,
                        bucket_idx,
                    )

                x_, y_, fuse_, explode_state_, bucket_explode_state_, bucket_idx_ = jax.lax.cond(
                    (~bombs_should_explode_) & (bucket_explode_state == self.consts.DEFAULT_STATE),
                    fall,
                    lambda: (x, y, fuse, explode_state, bucket_explode_state, bucket_idx),
                )

                done = (
                    (explode_state_ >= self.consts.BOMB_EXPLODE_STATES)
                    | (bucket_explode_state_ >= self.consts.BOMB_BUCKET_EXPLODE_STATES)
                )
                bomb_ = jnp.array([
                    x_, y_, fuse_, bomb_[TYPE],
                    explode_state_, bucket_explode_state_,
                    bucket_idx_, jnp.where(done, 0, bomb_[ACTIVE]),
                    explodes_in,
                ])
                return bomb_, bombs_should_explode_, level_finished_, level_success_

            return jax.lax.cond(
                bomb[ACTIVE] == 1,
                active_case,
                lambda operand: operand[:4],
                operand=(bomb, bombs_should_explode, level_finished, level_success, bomb_key),
            )

        def update_bombs_when_active(operand):
            bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key = operand

            bombs, bombs_dropped, key, level_finished, level_success = maybe_spawn_bomb(
                bombs, bombs_dropped, level_finished, level_success, key
            )

            bombs, score_deltas = jax.vmap(update_bucket_collision)(bombs)
            score = score + jnp.sum(score_deltas)

            bombs, bombs_should_explode, level_finished, level_success = trigger_ground_explosion(
                bombs, bombs_should_explode, level_finished, level_success
            )

            split_keys = jax.random.split(key, bombs.shape[0] + 1)
            key = split_keys[0]
            bomb_keys = split_keys[1:]

            bombs, all_bombs_should_explode, all_level_finished, all_level_success = jax.vmap(advance_bomb_state)(bombs, bomb_keys)
            bombs_should_explode = bombs_should_explode | jnp.any(all_bombs_should_explode)
            level_finished = level_finished | jnp.any(all_level_finished)
            level_success = level_success | jnp.any(all_level_success)

            return bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key

        return jax.lax.cond(
            bombs_falling_and_exploding,
            update_bombs_when_active,
            lambda x: x,
            (bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key),
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_background_step(self, bombs, buckets_pos, bombs_should_explode, background_flickering, background_state, bombs_dropped, bombs_falling_and_exploding,
                               waiting_for_restart, level_finished, level_success, level, lives):
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

        def enter_waiting_for_restart():
            # deactivate the topmost active bucket
            active_mask = buckets_pos[:, 2] == 1
            remove_idx = jnp.argmax(active_mask)

            new_buckets = buckets_pos.at[remove_idx, 2].set(0)

            new_level = jnp.where(level > 1, level - 1, level)
            return (
                0,  # bombs_dropped
                False,  # bombs_falling_and_exploding
                True,  # waiting_for_restart
                True,  # level_finished
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
            waiting_for_restart,
            level_finished,
            level_success,
            background_flickering,
            background_state,
            lives,
            level,
            buckets_pos,
        ) = jax.lax.cond(
            failure_step & animation_done,
            enter_waiting_for_restart,
            lambda: (
                bombs_dropped,
                bombs_falling_and_exploding,
                waiting_for_restart,
                level_finished,
                level_success,
                background_flickering,
                background_state,
                lives,
                level,
                buckets_pos,
            ),
        )

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

        return bombs_should_explode, background_flickering, background_state, bombs_dropped, bombs_falling_and_exploding, waiting_for_restart, level_finished, level_success, level, lives, buckets_pos

    @partial(jax.jit, static_argnums=(0,))
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
            mad_bomber_pos_x >= self.consts.BUCKET_MAX_ALLOWED_POS_X,
            True,
            mad_bomber_going_left,
        )

        mad_bomber_going_left = jnp.where(
            mad_bomber_pos_x <= self.consts.BUCKET_MIN_ALLOWED_POS_X,
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
        restart_pressed = state.waiting_for_restart & (action == self.action_set[2])

        def restart_after_failure():
            return KaboomState(
                mad_bomber_pos_x=state.mad_bomber_pos_x,
                mad_bomber_pos_y=state.mad_bomber_pos_y,
                mad_bomber_going_left=state.mad_bomber_going_left,
                mad_bomber_motion_counter=state.mad_bomber_motion_counter,
                bombs_states=state.bombs_states,
                buckets_pos=state.buckets_pos,
                buckets_jitter_state=state.buckets_jitter_state,
                buckets_moving_state=state.buckets_moving_state,
                buckets_were_moving_right=state.buckets_were_moving_right,
                score=state.score,
                lives=state.lives,
                level=state.level,
                bombs_dropped=jnp.array(0),
                frames_counter=state.frames_counter + 1,
                bombs_falling_and_exploding=True,
                bombs_exploding=False,
                background_flickering=False,
                background_state=jnp.array(self.consts.DEFAULT_STATE, dtype=jnp.int32),
                waiting_for_restart=False,
                level_finished=False,
                level_success=False,
                key=state.key,
            )

        def wait_on_failure():
            return state.replace(frames_counter=state.frames_counter + 1)

        def normal_step():
            buckets_pos, buckets_moving_state, buckets_were_moving_right, buckets_jitter_state, frames_counter, key \
                = self.update_bucket_step(state, action, state.buckets_pos, state.buckets_moving_state,
                                          state.buckets_were_moving_right, state.buckets_jitter_state, state.frames_counter,
                                          state.key)

            bombs, buckets_pos, mad_bomber_pos, level, bombs_dropped, score, level_finished, level_success, frames_counter, bombs_falling_and_exploding, bombs_should_explode, key \
                = self.update_bombs_step(state.bombs_states, buckets_pos,
                                         (state.mad_bomber_pos_x, state.mad_bomber_pos_y), state.level, state.bombs_dropped,
                                         state.score, state.level_finished, state.level_success, frames_counter,
                                         state.bombs_falling_and_exploding, state.bombs_exploding, key)

            bombs_should_explode, background_flickering, background_state, bombs_dropped, bombs_falling_and_exploding, waiting_for_restart, level_finished, level_success, level, lives, buckets_pos \
                = self.update_background_step(bombs, buckets_pos, bombs_should_explode, state.background_flickering, state.background_state, bombs_dropped, bombs_falling_and_exploding,
                                   state.waiting_for_restart, level_finished, level_success, level, state.lives)

            mad_bomber_motion_counter, mad_bomber_pos_x, mad_bomber_going_left, key \
                = self.update_mad_bomber_step(state.mad_bomber_motion_counter, mad_bomber_pos[0],
                                              state.mad_bomber_going_left, level, level_finished, bombs_dropped, key)

            frames_counter += 1
            return KaboomState(
                mad_bomber_pos_x=mad_bomber_pos_x,
                mad_bomber_pos_y=mad_bomber_pos[1],
                mad_bomber_going_left=mad_bomber_going_left,
                mad_bomber_motion_counter=mad_bomber_motion_counter,
                bombs_states=bombs,
                buckets_pos=buckets_pos,
                buckets_jitter_state=buckets_jitter_state,
                buckets_moving_state=buckets_moving_state,
                buckets_were_moving_right=buckets_were_moving_right,
                score=score,
                lives=lives,
                level=level,
                bombs_dropped=bombs_dropped,
                frames_counter=frames_counter,
                bombs_falling_and_exploding=bombs_falling_and_exploding,
                bombs_exploding=bombs_should_explode,
                background_flickering=background_flickering,
                background_state=background_state,
                waiting_for_restart=waiting_for_restart,
                level_finished=level_finished,
                level_success=level_success,
                key=key
            )

        new_state = jax.lax.cond(
            state.waiting_for_restart,
            lambda _: jax.lax.cond(restart_pressed, lambda __: restart_after_failure(), lambda __: wait_on_failure(), operand=None),
            lambda _: normal_step(),
            operand=None,
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

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "mad_bomber_pos": spaces.Box(
                    low=0, high=160, shape=(2,), dtype=jnp.int32
                ),
                "buckets_pos": spaces.Box(
                    low=-1, high=210, shape=(3, 3), dtype=jnp.int32
                ),
                "bombs": spaces.Box(
                    low=-1,
                    high=210,
                    shape=(15, 9),
                    dtype=jnp.int32,
                ),
                "score": spaces.Box(
                    low=0, high=1_000_000, shape=(), dtype=jnp.int32
                ),
                "lives": spaces.Box(
                    low=0, high=3, shape=(), dtype=jnp.int32
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

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
    def __init__(self, consts: KaboomConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or KaboomConstants()
        super().__init__(self.consts)

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/kaboom"

        # 2. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)

        # Pre-stack all related sprites for easy indexing in the render loop
        self.BACKGROUNDS = [self.SHAPE_MASKS['bg_top'], self.SHAPE_MASKS['bg_bottom']]
        self.BOMBS = self.SHAPE_MASKS['bombs']
        self.BOMB_FUSE_STATES = self.SHAPE_MASKS['bomb_fuse_states']
        self.BOMB_EXPLODE_STATES = self.SHAPE_MASKS['bomb_explode_states']
        self.BOMB_BUCKET_EXPLODE_STATES = self.SHAPE_MASKS['bomb_bucket_explode_states']
        self.SCORES = self.SHAPE_MASKS['scores']

        # Create randomized backgrounds
        self.BACKGROUND_TOP_VARIANTS, self.BACKGROUND_BOTTOM_VARIANTS = self._create_random_backgrounds()

    def _create_random_backgrounds(self, n_variants: int = 32, key=None):
        """
        Create separate stacks of random solid-color top and bottom backgrounds.

        The top and bottom masks have different shapes, so they must be generated
        and stored independently to stay compatible with JAX shape requirements.

        Returns:
            tuple[jnp.Array, jnp.Array]
            - top variants: (n_variants, H_top, W_top)
            - bottom variants: (n_variants, H_bottom, W_bottom)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        top_h, top_w = self.BACKGROUNDS[0].shape
        bottom_h, bottom_w = self.BACKGROUNDS[1].shape

        top_variants = []
        bottom_variants = []

        for _ in range(n_variants):
            key, subkey_top, subkey_bottom = jax.random.split(key, 3)

            top_color = jax.random.randint(subkey_top, shape=(), minval=0, maxval=19, dtype=jnp.uint8)
            bottom_color = jax.random.randint(subkey_bottom, shape=(), minval=0, maxval=19, dtype=jnp.uint8)

            top_variants.append(jnp.full((top_h, top_w), top_color, dtype=jnp.uint8))
            bottom_variants.append(jnp.full((bottom_h, bottom_w), bottom_color, dtype=jnp.uint8))

        return jnp.stack(top_variants, axis=0), jnp.stack(bottom_variants, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KaboomState) -> chex.Array:
        # Background_border
        raster = self.jr.create_object_raster(self.BACKGROUND)

        top_flicker_idx = jnp.mod(state.background_state, self.consts.BACKGROUND_STATES)
        bottom_flicker_idx = jnp.mod(state.background_state, self.consts.BACKGROUND_STATES)
        background_top = jax.lax.cond(
            state.background_flickering,
            lambda _: self.BACKGROUND_TOP_VARIANTS[top_flicker_idx],
            lambda _: self.BACKGROUNDS[0],
            operand=None,
        )
        background_bottom = jax.lax.cond(
            state.background_flickering,
            lambda _: self.BACKGROUND_BOTTOM_VARIANTS[bottom_flicker_idx],
            lambda _: self.BACKGROUNDS[1],
            operand=None,
        )

        # Background_top
        raster = self.jr.render_at(
            raster,
            x=self.consts.BACKGROUND_TOP_POS[0],
            y=self.consts.BACKGROUND_TOP_POS[1],
            sprite_mask=background_top,
        )

        # Background_bottom
        raster = self.jr.render_at(
            raster,
            x=self.consts.BACKGROUND_BOTTOM_POS[0],
            y=self.consts.BACKGROUND_BOTTOM_POS[1],
            sprite_mask=background_bottom,
        )

        # Score
        score = state.score.astype(jnp.int32)

        def draw_score_digit(i, raster_):
            pow10 = jnp.asarray(10, dtype=jnp.int32) ** i
            digit_value = (score // pow10) % 10
            x_pos = self.consts.SCORE_POS[0] - 7 * i
            should_render = (i == 0) | (score >= pow10)
            digit = self.jr.int_to_digits(digit_value, max_digits=1)

            return jax.lax.cond(
                should_render,
                lambda r: self.jr.render_label(
                    r,
                    x_pos,
                    self.consts.SCORE_POS[1],
                    digit,
                    self.SCORES,
                    spacing=-7,
                    max_digits=1,
                ),
                lambda r: r,
                raster_,
            )

        raster = jax.lax.fori_loop(0, 7, draw_score_digit, raster)

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

            def draw_active(r2):
                x = bomb[0].astype(int)
                y = bomb[1].astype(int)

                def draw_air_explosion(r3):
                    idx = (bomb[4] > 3).astype(jnp.int32) + (bomb[4] > 7).astype(jnp.int32)
                    return self.jr.render_at(r3, x, y, self.BOMB_EXPLODE_STATES[idx])

                def draw_bucket_explosion(r3):
                    idx = (bomb[5] > 3).astype(jnp.int32) + (bomb[5] > 7).astype(jnp.int32)
                    bucket = state.buckets_pos[bomb[6]]
                    by = bucket[1] - self.BOMB_BUCKET_EXPLODE_STATES[idx].shape[0]
                    return self.jr.render_at(r3, bucket[0], by, self.BOMB_BUCKET_EXPLODE_STATES[idx])

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