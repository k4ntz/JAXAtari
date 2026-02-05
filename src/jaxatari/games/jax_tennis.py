import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional, Sequence, Callable
import jax.lax
import jax.numpy as jnp
import chex
import jax.random as random
import jax
import numpy as np
import collections
from flax import struct

# Replaced legacy utils with new one
import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import AutoDerivedConstants

rand_key = random.key(0)


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Tennis.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    NOTE: Background and procedural swatches are handled in the renderer init.
    """
    return (
        {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
        {'name': 'ball_shadow', 'type': 'single', 'file': 'ball_shadow.npy'},
        {'name': 'player_anim', 'type': 'group', 'files': [
            'player_no_racket.npy',
            'player_no_racket_1.npy',
            'player_no_racket_2.npy',
            'player_no_racket_3.npy'
        ]},
        {'name': 'racket_anim', 'type': 'group', 'files': [
            'racket_red_0.npy',
            'racket_red_1.npy',
            'racket_red_2.npy',
            'racket_red_3.npy'
        ]},
        {'name': 'digits', 'type': 'digits', 'pattern': 'ui_blue_{}.npy'},
        {'name': 'ui_a', 'type': 'single', 'file': 'ui_blue_9.npy'},
    )


class TennisConstants(AutoDerivedConstants):
    # frame (window) constants
    FRAME_WIDTH: int = struct.field(pytree_node=False, default=160)
    FRAME_HEIGHT: int = struct.field(pytree_node=False, default=210)

    # field constants (the actual tennis court)
    FIELD_WIDTH_TOP: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(79))
    FIELD_WIDTH_BOTTOM: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(111))
    FIELD_HEIGHT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(130))

    # game constants (these values are used for the actual gameplay calculations)
    GAME_OFFSET_LEFT_BOTTOM: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(15 + 1))
    GAME_OFFSET_TOP: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(43.0))
    GAME_WIDTH: chex.Array = struct.field(pytree_node=False, default=None)
    GAME_HEIGHT: chex.Array = struct.field(pytree_node=False, default=None)
    GAME_MIDDLE: chex.Array = struct.field(pytree_node=False, default=None)
    GAME_OFFSET_BOTTOM: chex.Array = struct.field(pytree_node=False, default=None)
    PAUSE_DURATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(100))

    # player constants
    PLAYER_CONST: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(0))
    PLAYER_WIDTH: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(13))
    PLAYER_HEIGHT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(23))
    PLAYER_MIN_X: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(8))
    PLAYER_MAX_X: chex.Array = struct.field(pytree_node=False, default_factory=lambda:jnp.array(144))
    PLAYER_Y_LOWER_BOUND_BOTTOM: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(160))
    PLAYER_Y_UPPER_BOUND_BOTTOM: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(109))
    PLAYER_Y_LOWER_BOUND_TOP: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(70))
    PLAYER_Y_UPPER_BOUND_TOP: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(15))

    START_X: chex.Array = struct.field(pytree_node=False, default=None)
    PLAYER_START_Y: chex.Array = struct.field(pytree_node=False, default=None)
    ENEMY_START_Y: chex.Array = struct.field(pytree_node=False, default=None)

    PLAYER_START_DIRECTION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))
    PLAYER_START_FIELD: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))

    # ball constants
    BALL_GRAVITY_PER_FRAME: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1.1))
    BALL_SERVING_BOUNCE_VELOCITY_BASE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(21))
    BALL_SERVING_BOUNCE_VELOCITY_RANDOM_OFFSET: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))
    BALL_WIDTH: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(2.0))
    LONG_HIT_THRESHOLD_TOP: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(52))
    LONG_HIT_THRESHOLD_BOTTOM: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(121))
    LONG_HIT_DISTANCE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(91))
    SHORT_HIT_DISTANCE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(40))

    # enemy constants
    ENEMY_CONST: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))

    GAME_MIDDLE_HORIZONTAL: chex.Array = struct.field(pytree_node=False, default=None)
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default=_get_default_asset_config())

    def compute_derived(self) -> Dict[str, Any]:
        game_width = self.FIELD_WIDTH_BOTTOM
        game_height = self.FIELD_HEIGHT
        game_middle = self.GAME_OFFSET_TOP + 0.5 * game_height
        game_offset_bottom = self.GAME_OFFSET_TOP + game_height
        return {
            'GAME_WIDTH': game_width,
            'GAME_HEIGHT': game_height,
            'GAME_MIDDLE': game_middle,
            'GAME_OFFSET_BOTTOM': game_offset_bottom,
            'START_X': self.GAME_OFFSET_LEFT_BOTTOM + 0.5 * game_width - 0.5 * self.PLAYER_WIDTH,
            'PLAYER_START_Y': self.GAME_OFFSET_TOP - self.PLAYER_HEIGHT,
            'ENEMY_START_Y': game_offset_bottom - self.PLAYER_HEIGHT,
            'GAME_MIDDLE_HORIZONTAL': (self.FRAME_WIDTH - self.PLAYER_WIDTH) / 2,
        }




@struct.dataclass
class BallState:
    ball_x: chex.Array  # x-coordinate of the ball
    ball_y: chex.Array  # y-coordinate of the ball
    ball_z: chex.Array  # z-coordinate of the ball
    ball_z_fp: chex.Array  # z-coordinate of the ball with exactly one point (effectively ball_z * 10, used for calculations)
    ball_velocity_z_fp: chex.Array  # z-velocity of the ball with exactly one point
    ball_hit_start_x: chex.Array  # x-coordinate of the location where the ball was last hit
    ball_hit_start_y: chex.Array  # y-coordinate of the location where the ball was last hit
    ball_hit_target_x: chex.Array  # x-coordinate of the location where the ball was last aimed towards
    ball_hit_target_y: chex.Array  # y-coordinate of the location where the ball was last aimed towards
    move_x: chex.Array  # Normalized distance from ball_x to ball_hit_target_x, updated on hit
    move_y: chex.Array  # Normalized distance from ball_y to ball_hit_target_y, updated on hit
    bounces: chex.Array  # how many times the ball has hit the ground since it was last hit by an entity
    last_hit: chex.Array  # 0 if last hit was performed by player, 1 if last hit was by enemy


@struct.dataclass
class PlayerState:
    player_x: chex.Array  # x-coordinate of the player
    player_y: chex.Array  # y-coordinate of the player
    player_direction: chex.Array  # direction the player is currently facing in (-1: look towards left, 1: look towards right)
    player_field: chex.Array  # top (1) or bottom (-1) field
    player_serving: chex.Array  # true: Player is serving, false: Enemy is serving
    player_walk_speed: chex.Array


@struct.dataclass
class EnemyState:
    enemy_x: chex.Array  # x-coordinate of the enemy
    enemy_y: chex.Array  # y-coordinate of the enemy
    prev_walking_direction: chex.Array  # previous walking direction (in x-direction) of the enemy, -1 = towards x=min, 1 = towards x=max
    enemy_direction: chex.Array
    y_movement_direction: chex.Array  # direction in which the enemy is moving until it hits the ball, -1 towards net 1 away from net


@struct.dataclass
class GameState:
    is_serving: chex.Array  # whether the game is currently in serving state (ball bouncing on one side until player hits)
    pause_counter: chex.Array  # delay between restart of game
    player_score: chex.Array  # The score line within the current set (goes up in increments of 1, instead of traditional tennis counting)
    enemy_score: chex.Array
    player_game_score: chex.Array  # Number of won sets
    enemy_game_score: chex.Array
    is_finished: chex.Array  # True if the game is finished (Player or enemy has won the game)


@struct.dataclass
class AnimatorState:
    player_frame: chex.Array = 0
    enemy_frame: chex.Array = 0
    player_racket_frame: chex.Array = 0
    player_racket_animation: chex.Array = False
    enemy_racket_frame: chex.Array = 0
    enemy_racket_animation: chex.Array = False


@struct.dataclass
class TennisState:
    player_state: PlayerState
    enemy_state: EnemyState
    ball_state: BallState
    game_state: GameState
    counter: chex.Array
    animator_state: AnimatorState
    random_key: jax.random.PRNGKey


# All top-level functions (tennis_step, normal_step, animator_step, 
# check_score, check_set, check_end, player_step, enemy_step, 
# ball_step, is_overlapping, handle_ball_fire, tennis_reset)
# have been moved into the TennisJaxEnv class below.


@struct.dataclass
class PlayerObs:
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_field: chex.Array  # top (1) or bottom (-1) field
    player_serving: chex.Array  # true: Player is serving, false: Enemy is serving


@struct.dataclass
class EnemyObs:
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_direction: chex.Array


@struct.dataclass
class BallObs:
    ball_x: chex.Array  # x-coordinate of the ball
    ball_y: chex.Array  # y-coordinate of the ball
    ball_z: chex.Array  # z-coordinate of the ball
    bounces: chex.Array  # how many times the ball has hit the ground since it was last hit by an entity
    last_hit: chex.Array  # 0 if last hit was performed by player, 1 if last hit was by enemy


@struct.dataclass
class TennisObs:
    player: PlayerObs
    enemy: EnemyObs
    ball: BallObs
    is_serving_state: chex.Array  # whether the game is currently in serving state (ball bouncing on one side until player hits)
    player_points: chex.Array  # The score line within the current set (goes up in increments of 1, instead of traditional tennis counting)
    enemy_points: chex.Array
    player_sets: chex.Array  # Number of won sets
    enemy_sets: chex.Array


@struct.dataclass
class TennisInfo:
    pass 


class TennisJaxEnv(JaxEnvironment[TennisState, TennisObs, TennisInfo, TennisConstants]):
    # Minimal ALE action set for Tennis (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: TennisConstants = None):
        if consts is None:
            consts = TennisConstants()
        super().__init__(consts)
        self.renderer = TennisRenderer(consts)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[TennisObs, TennisState]:
        """
        Provides the initial state for the game. For that purpose, we use the default values assigned in TennisState.

    Returns:
            TennisState: The initial state of the game.
        """
        reset_state = TennisState(
            player_state=PlayerState(
                player_x = jnp.array(self.consts.START_X),
                player_y = jnp.array(self.consts.PLAYER_START_Y),
                player_direction=jnp.array(self.consts.PLAYER_START_DIRECTION),
                player_field=jnp.array(self.consts.PLAYER_START_FIELD),
                player_serving=jnp.array(True),
                player_walk_speed=jnp.array(1.0),
            ),
            enemy_state=EnemyState(
                enemy_x=jnp.array(self.consts.START_X),
                enemy_y=jnp.array(self.consts.ENEMY_START_Y),
                prev_walking_direction=jnp.array(0.0),
                enemy_direction=jnp.array(self.consts.PLAYER_START_DIRECTION),
                y_movement_direction=jnp.array(1),
            ),
            ball_state=BallState(
                ball_x=jnp.array(self.consts.FRAME_WIDTH / 2.0 - self.consts.BALL_WIDTH / 2),
                ball_y=jnp.array(self.consts.GAME_OFFSET_TOP),
                ball_z=jnp.array(0.0),
                ball_z_fp=jnp.array(0.0),
                ball_velocity_z_fp=jnp.array(0.0),
                ball_hit_start_x=jnp.array(0.0),
                ball_hit_start_y=jnp.array(0.0),
                ball_hit_target_x=jnp.array(self.consts.FRAME_WIDTH / 2.0 - self.consts.BALL_WIDTH / 2),
                ball_hit_target_y=jnp.array(self.consts.GAME_OFFSET_TOP),
                move_x=jnp.array(0.0),
                move_y=jnp.array(0.0),
                bounces=jnp.array(0),
                last_hit=jnp.array(-1)
            ),
            game_state=GameState(
                is_serving=jnp.array(True),
                pause_counter=jnp.array(0),
                player_score=jnp.array(0),
                enemy_score=jnp.array(0),
                player_game_score=jnp.array(0),
                enemy_game_score=jnp.array(0),
                is_finished=jnp.array(False),
            ),
            counter=jnp.array(0),
            animator_state = AnimatorState(),
            random_key = jax.random.PRNGKey(0)
        )
        reset_obs = self._get_observation(reset_state)

        return reset_obs, reset_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TennisState, action) -> Tuple[TennisObs, TennisState, float, bool, TennisInfo]:
        """
        Updates the entire state of the game by calling all step functions.

        Args:
            state (TennisState): The current state of the game.
            action: The action to apply.

        Returns:
            TennisState: The updated state of the game.
        """
        
        def _pause_step(state: TennisState) -> TennisState:
            return TennisState(state.player_state, state.enemy_state,
                                state.ball_state,
                                GameState(
                                    state.game_state.is_serving,
                                    state.game_state.pause_counter - 1,
                                    state.game_state.player_score,
                                    state.game_state.enemy_score,
                                    state.game_state.player_game_score,
                                    state.game_state.enemy_game_score,
                                    state.game_state.is_finished,
                                ),
                                state.counter + 1, animator_state=AnimatorState(),random_key=state.random_key)
        
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        
        new_state = jax.lax.cond(state.game_state.is_finished,
                        lambda s: s,
                        lambda s: jax.lax.cond(s.game_state.pause_counter > 0,
                                               _pause_step,
                                               lambda s_inner: self._normal_step(s_inner, atari_action),
                                               s
                                               ),
                        state
                        )
        new_obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _normal_step(self, state: TennisState, atari_action) -> TennisState:
        """
        Updates the entire state of the game by calling all step functions. Should be used when game is not paused.

        Args:
            state (TennisState): The current state of the game.
            atari_action: The translated ALE action to apply.
        Returns:
            TennisState: The updated state of the game.
        """
        new_state_after_score_check = self._check_score(state)
        new_state_after_ball_step = self._ball_step(new_state_after_score_check, atari_action)
        new_player_state = self._player_step(new_state_after_ball_step, atari_action)
        new_enemy_state = self._enemy_step(new_state_after_ball_step)
        new_animator_state = self._animator_step(state, new_player_state, new_enemy_state)

        return TennisState(new_player_state, new_enemy_state, new_state_after_ball_step.ball_state,
                           new_state_after_ball_step.game_state, state.counter + 1, new_animator_state, state.random_key)

    @partial(jax.jit, static_argnums=(0,))
    def _animator_step(self, state: TennisState, new_player_state, new_enemy_state) -> AnimatorState:
        """
        Updates the animator state depending on the current player and enemy state.

        Args:
            state (TennisState): The current state of the game.
            new_player_state (PlayerState): The new player state, recorded 1 frame after `state`.
            new_enemy_state (EnemyState): The enew enemy state, recorded 1 frame after `state`.

        Returns:
            AnimatorState: The updated animator state.
        """

        @jax.jit
        def check_has_moved(prev_x, cur_x, prev_y, cur_y):
            return jnp.logical_or(
                prev_x != cur_x,
                prev_y != cur_y,
            )

        # Animations for player
        has_player_moved = check_has_moved(state.player_state.player_x, new_player_state.player_x,
                                           state.player_state.player_y, new_player_state.player_y)


@jax.jit
def check_end(state: GameState) -> GameState:
    """
    Checks whether the entire game is over and updates the score accordingly.

    Args:
        state (GameState): The current state of the game.

    Returns:
        GameState: The updated state of the game.
    """

    player_won = jnp.logical_and(state.player_game_score >= 6, state.player_game_score >= state.enemy_game_score + 2)
    enemy_won = jnp.logical_and(state.enemy_game_score >= 6, state.enemy_game_score >= state.player_game_score + 2)
    is_finished = jnp.where(jnp.logical_or(player_won, enemy_won), True, False)
    return GameState(state.is_serving, state.pause_counter, state.player_score, state.enemy_score,
                     state.player_game_score, state.enemy_game_score, is_finished)


@jax.jit
def player_step(state: TennisState, action: chex.Array, consts) -> PlayerState:
    """
    Updates player position based on provided action and applies bounding box.

    Args:
        state (PlayerState): The current player state.
        action (chex.Array): The action to apply.

    Returns:
        PlayerState: The updated player state.
    """

    player_state = state.player_state

    # does the action contain UP
    up = jnp.any(
        jnp.array(
            [action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT,
             action == Action.UPFIRE, action == Action.UPRIGHTFIRE,
             action == Action.UPLEFTFIRE]
        )
    )
    # does the action contain DOWN
    down = jnp.any(
        jnp.array(
            [action == Action.DOWN, action == Action.DOWNRIGHT, action == Action.DOWNLEFT,
             action == Action.DOWNFIRE, action == Action.DOWNRIGHTFIRE,
             action == Action.DOWNLEFTFIRE]
        )
    )
    # does the action contain LEFT
    left = jnp.any(
        jnp.array(
            [action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT,
             action == Action.LEFTFIRE, action == Action.UPLEFTFIRE,
             action == Action.DOWNLEFTFIRE]
        )
    )
    # does the action contain RIGHT
    right = jnp.any(
        jnp.array(
            [action == Action.RIGHT, action == Action.UPRIGHT, action == Action.DOWNRIGHT,
             action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE,
             action == Action.DOWNRIGHTFIRE]
        )
    )

    # move left if the player is trying to move left
    player_x = jnp.where(
        left,
        player_state.player_x - player_state.player_walk_speed,
        player_state.player_x,
    )
    # move right if the player is trying to move right
    player_x = jnp.where(
        right,
        player_state.player_x + player_state.player_walk_speed,
        player_x,
    )
    # apply X bounding box
    player_x = jnp.clip(player_x, consts.PLAYER_MIN_X, consts.PLAYER_MAX_X)

    # move up if the player is trying to move up
    player_y = jnp.where(
        jnp.logical_and(
            up,
            jnp.logical_not(state.game_state.is_serving)  # not allowed to change y position while someone is serving
        ),
        player_state.player_y - player_state.player_walk_speed,
        player_state.player_y,
    )

    # move down if the player is trying to move down
    player_y = jnp.where(
        jnp.logical_and(
            down,
            jnp.logical_not(state.game_state.is_serving)  # not allowed to change y position while someone is serving
        ),
        player_state.player_y + player_state.player_walk_speed,
        player_y,
    )
    # apply Y bounding box
    player_y = jnp.where(
        player_state.player_field == 1,
        jnp.clip(player_y, consts.PLAYER_Y_UPPER_BOUND_TOP, consts.PLAYER_Y_LOWER_BOUND_TOP),
        jnp.clip(player_y, consts.PLAYER_Y_UPPER_BOUND_BOTTOM, consts.PLAYER_Y_LOWER_BOUND_BOTTOM)
    )

    new_player_direction = jnp.where(
        player_state.player_x > state.ball_state.ball_x,
        -1,
        player_state.player_direction
    )

    new_player_direction = jnp.where(
        player_state.player_x < state.ball_state.ball_x,
        1,
        new_player_direction
    )

    return PlayerState(
        player_x,
        player_y,
        new_player_direction,
        player_state.player_field,
        player_state.player_serving,
        player_state.player_walk_speed,
    )


@jax.jit
def enemy_step(state: TennisState, consts) -> EnemyState:
    """
    Updates enemy position by following the enemy strategy explained below.

    Enemy strategy:
        x coordinate:
            - just follow x of the ball
            - keep ball at center of sprite so that sprite does not flip often
        y coordinate:
            - rush towards net after hitting ball
            - stay at net for some time
            - turning point (does not quite line up with player hitting ball, usually slightly earlier)
            - move as far away from net as possible
            - after reaching limit sometimes starts moving towards net again

    Args:
        state (TennisState): The current state of the game.

    Returns:
        EnemyState: The updated enemy state.
    """

    # x-coordinate
    enemy_hit_offset = state.enemy_state.prev_walking_direction * 5 * -1 * 0  # this is always 0
    enemy_x_hit_point = state.enemy_state.enemy_x + consts.PLAYER_WIDTH / 2 + enemy_hit_offset

    player_x_hit_point = state.player_state.player_x + consts.PLAYER_WIDTH / 2
    ball_tracking_tolerance = 1
    x_tracking_tolerance = 2

    # move to middle
    def move_x_to_middle():
        middle_step_x = jnp.where(jnp.less_equal(state.enemy_state.enemy_x, consts.GAME_MIDDLE_HORIZONTAL),
                                  state.enemy_state.enemy_x + 1,
                                  state.enemy_state.enemy_x - 1)
        return jnp.where(jnp.abs(state.enemy_state.enemy_x - consts.GAME_MIDDLE_HORIZONTAL) > 1, middle_step_x,
                         state.enemy_state.enemy_x)

    def track_ball_x():
        enemy_aiming_x_offset = jnp.where(
            player_x_hit_point < consts.FRAME_WIDTH / 2,
            5,
            -15
        )
        diff = state.ball_state.ball_x - (enemy_x_hit_point + enemy_aiming_x_offset)

        # move right if ball is sufficiently to the right
        new_enemy_x = jnp.where(
            diff > x_tracking_tolerance,
            state.enemy_state.enemy_x + 1,
            state.enemy_state.enemy_x
        )

        # move left if ball is sufficiently to the left
        new_enemy_x = jnp.where(
            diff < -x_tracking_tolerance,
            state.enemy_state.enemy_x - 1,
            new_enemy_x
        )
        return new_enemy_x

    new_enemy_x = jax.lax.cond(state.ball_state.last_hit == 1, move_x_to_middle, track_ball_x)

    cur_walking_direction = jnp.where(
        new_enemy_x - state.enemy_state.enemy_x < 0,
        -1,
        state.enemy_state.prev_walking_direction
    )
    cur_walking_direction = jnp.where(
        new_enemy_x - state.enemy_state.enemy_x > 0,
        1,
        cur_walking_direction
    )

    # jnp.clip(new_enemy_x - state.enemy_state.enemy_x, -1, 1))

    should_perform_direction_change = jnp.logical_or(jnp.logical_or(
        jnp.abs((enemy_x_hit_point) - state.ball_state.ball_x) >= ball_tracking_tolerance,
        # state.enemy_state.prev_walking_direction == cur_walking_direction
        False
    ), state.ball_state.last_hit == 1)

    new_enemy_x = jnp.where(should_perform_direction_change, new_enemy_x, state.enemy_state.enemy_x)

    # y-coordinate

    def enemy_y_step():
        state_after_y = jax.lax.cond(
            state.ball_state.last_hit == 1,
            # last hit was enemy rush net
            lambda _: EnemyState(state.enemy_state.enemy_x,
                                 jnp.where(jnp.logical_and(state.enemy_state.enemy_y != consts.PLAYER_Y_LOWER_BOUND_TOP,
                                                           state.enemy_state.enemy_y != consts.PLAYER_Y_UPPER_BOUND_BOTTOM),
                                           state.enemy_state.enemy_y - state.player_state.player_field,
                                           state.enemy_state.enemy_y
                                           ), state.enemy_state.prev_walking_direction,
                                 state.enemy_state.enemy_direction, jnp.array(1)),
            # last hit was player move away from net until baseline is hit then move towards ball
            lambda _: EnemyState(state.enemy_state.enemy_x,
                                 jnp.where(state.player_state.player_field == 1,
                                           jnp.clip(state.enemy_state.enemy_y +
                                                          state.player_state.player_field * state.enemy_state.y_movement_direction,
                                                          consts.PLAYER_Y_UPPER_BOUND_BOTTOM,
                                                          consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                           jnp.clip(state.enemy_state.enemy_y +
                                                          state.player_state.player_field * state.enemy_state.y_movement_direction,
                                                          consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                          consts.PLAYER_Y_LOWER_BOUND_TOP)),
                                 state.enemy_state.prev_walking_direction,
                                 state.enemy_state.enemy_direction,
                                 jnp.where(jnp.logical_or(state.enemy_state.enemy_y == consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                          state.enemy_state.enemy_y == consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                           jnp.array(-1),
                                           state.enemy_state.y_movement_direction)), operand=None)

        return jax.lax.cond(
            state.game_state.is_serving,
            lambda _: state.enemy_state,
            lambda _: state_after_y,
            operand=None
        )

    enemy_state_after_y_step = enemy_y_step()

    new_enemy_direction = jnp.where(
        state.enemy_state.enemy_x > state.ball_state.ball_x,
        -1,
        state.enemy_state.enemy_direction
    )

    new_enemy_direction = jnp.where(
        state.enemy_state.enemy_x < state.ball_state.ball_x,
        1,
        new_enemy_direction
    )

    return EnemyState(
        new_enemy_x,
        enemy_state_after_y_step.enemy_y,
        cur_walking_direction,
        new_enemy_direction,
        enemy_state_after_y_step.y_movement_direction
    )


@jax.jit
def ball_step(state: TennisState, action, consts) -> TennisState:
    """
    Updates ball position by applying velocity and gravity. Also handles player-ball collisions
    and fires the ball if the provided action contains FIRE.

    Args:
        state (TennisState): The current state of the game.
        action (chex.Array): The action to apply.

    Returns:
        BallState: The updated ball state.
    """

    @jax.jit
    def get_serving_bounce_velocity() -> int:
        """
        Applies a random offset to the base serving bounce velocity.

        Returns:
            int: The calculated bounce velocity
        """

        return consts.BALL_SERVING_BOUNCE_VELOCITY_BASE + random.uniform(state.random_key) * consts.BALL_SERVING_BOUNCE_VELOCITY_RANDOM_OFFSET

    ball_state = state.ball_state

    # update the fixed-point z velocity value by applying either upward velocity or gravity
    new_ball_velocity_z_fp = jnp.where(
        ball_state.ball_z == 0,
        get_serving_bounce_velocity(),
        ball_state.ball_velocity_z_fp - consts.BALL_GRAVITY_PER_FRAME
    )

    # update the fixed-point z value by applying current velocity (can be positive or negative)
    new_ball_z_fp = ball_state.ball_z_fp + new_ball_velocity_z_fp
    # calculate actual z value using floor division by 10, because fixed-point value has exactly one point
    new_ball_z = new_ball_z_fp // 10

    # apply lower bounding box (500 is effectively no MAX bound)
    new_ball_z = jnp.clip(new_ball_z, 0, 500)
    new_ball_z_fp = jnp.clip(new_ball_z_fp, 0, 500)

    # ball movement in x/y direction is linear, no velocity involved
    new_ball_x = ball_state.ball_x + ball_state.move_x
    new_ball_y = ball_state.ball_y + ball_state.move_y

    player_state = state.player_state
    enemy_state = state.enemy_state

    # check player-ball collisions
    player_end = player_state.player_x + consts.PLAYER_WIDTH
    ball_end = ball_state.ball_x + consts.BALL_WIDTH
    player_overlap_ball_x = jnp.logical_not(
        jnp.logical_or(
            player_end <= ball_state.ball_x,
            ball_end <= player_state.player_x
        )
    )

    # Correct X positions based on facing direction
    corrected_player_x = jnp.where(
        player_state.player_direction == 1,
        player_state.player_x,
        player_state.player_x - consts.PLAYER_WIDTH + 2
    )

    corrected_enemy_x = jnp.where(
        enemy_state.enemy_direction == 1,
        enemy_state.enemy_x,
        enemy_state.enemy_x - consts.PLAYER_WIDTH + 2
    )

    upper_entity_x = jnp.where(
        player_state.player_field == 1,
        corrected_player_x,
        corrected_enemy_x
    )
    upper_entity_y = jnp.where(
        player_state.player_field == 1,
        player_state.player_y,
        enemy_state.enemy_y
    )

    lower_entity_x = jnp.where(
        player_state.player_field == -1,
        corrected_player_x,
        corrected_enemy_x
    )
    lower_entity_y = jnp.where(
        player_state.player_field == -1,
        player_state.player_y,
        enemy_state.enemy_y
    )

    upper_entity_overlapping_ball = jnp.logical_and(
        is_overlapping(
            upper_entity_x,
            consts.PLAYER_WIDTH,
            0,  # this is z pos
            consts.PLAYER_HEIGHT,
            ball_state.ball_x,
            consts.BALL_WIDTH,
            0,  # the original game ignores height when checking collision
            consts.BALL_WIDTH  # could be renamed to to BALL_SIZE because ball is square
        ),
        jnp.absolute(upper_entity_y + consts.PLAYER_HEIGHT - ball_state.ball_y) <= 3
    )

    lower_entity_overlapping_ball = jnp.logical_and(
        is_overlapping(
            lower_entity_x,
            consts.PLAYER_WIDTH,
            0,  # this is z pos
            consts.PLAYER_HEIGHT,
            ball_state.ball_x,
            consts.BALL_WIDTH,
            0,  # the original game ignores height when checking collision
            consts.BALL_WIDTH  # could be renamed to to BALL_SIZE because ball is square
        ),
        jnp.absolute(lower_entity_y + consts.PLAYER_HEIGHT - ball_state.ball_y) <= 3
    )

    lower_entity_performed_last_hit = jnp.logical_or(
        jnp.logical_and(
            state.player_state.player_field == -1, state.ball_state.last_hit != consts.ENEMY_CONST
        ),
        jnp.logical_and(
            state.player_state.player_field == 1, state.ball_state.last_hit != consts.PLAYER_CONST
        )
    )

    upper_entity_performed_last_hit = jnp.logical_or(
        jnp.logical_and(
            state.player_state.player_field == 1, state.ball_state.last_hit != consts.ENEMY_CONST
        ),
        jnp.logical_and(
            state.player_state.player_field == -1, state.ball_state.last_hit != consts.PLAYER_CONST
        )
    )

    # check if fire is pressed
    fire = jnp.any(jnp.array(
        [action == Action.FIRE, action == Action.LEFTFIRE, action == Action.DOWNLEFTFIRE,
         action == Action.DOWNFIRE,
         action == Action.DOWNRIGHTFIRE, action == Action.RIGHTFIRE,
         action == Action.UPRIGHTFIRE, action == Action.UPFIRE,
         action == Action.UPLEFTFIRE]))

    any_entity_ready_to_fire = jnp.logical_or(
        jnp.logical_and(
            upper_entity_overlapping_ball,
            lower_entity_performed_last_hit
        ),
        jnp.logical_and(
            lower_entity_overlapping_ball,
            upper_entity_performed_last_hit
        )
    )

    should_hit = jnp.logical_and(any_entity_ready_to_fire, jnp.logical_or(
        jnp.logical_not(jnp.logical_and(state.player_state.player_serving, state.game_state.is_serving)),
        fire
    ))
    new_is_serving = jnp.where(should_hit, False, state.game_state.is_serving)

    # no need to check whether the lower entity is actually overlapping because this variable won't be used if it isn't
    ball_fire_direction = jnp.where(
        upper_entity_overlapping_ball,
        1,
        -1
    )
    # no need to check whether the lower entity is actually overlapping because this variable won't be used if it isn't
    hitting_entity_x = jnp.where(
        upper_entity_overlapping_ball,
        upper_entity_x,
        lower_entity_x
    )

    hitting_entity_y = jnp.where(
        upper_entity_overlapping_ball,
        upper_entity_y,
        lower_entity_y
    )

    # record which entity hit the ball most recently
    new_last_hit = jnp.where(should_hit,
                             # player hit
                             jnp.where(
                                 jnp.logical_or(
                                     jnp.logical_and(upper_entity_overlapping_ball, player_state.player_field == 1),
                                     jnp.logical_and(lower_entity_overlapping_ball, player_state.player_field == -1)
                                 ),
                                 0,
                                 1
                             ),
                             state.ball_state.last_hit
                             )

    ball_state_after_fire = jax.lax.cond(
        should_hit,
        lambda _: handle_ball_fire(state, hitting_entity_x, hitting_entity_y, ball_fire_direction, consts),
        lambda _: BallState(
            new_ball_x,
            new_ball_y,
            new_ball_z,
            new_ball_z_fp,
            new_ball_velocity_z_fp,
            ball_state.ball_hit_start_x,
            ball_state.ball_hit_start_y,
            ball_state.ball_hit_target_x,
            ball_state.ball_hit_target_y,
            ball_state.move_x,
            ball_state.move_y,
            ball_state.bounces,
            ball_state.last_hit,
        ),
        None
    )

    new_random_key, _ = jax.random.split(state.random_key)

    return TennisState(
        player_state,
        enemy_state,
        BallState(
            ball_state_after_fire.ball_x,
            ball_state_after_fire.ball_y,
            ball_state_after_fire.ball_z,
            ball_state_after_fire.ball_z_fp,
            ball_state_after_fire.ball_velocity_z_fp,
            ball_state_after_fire.ball_hit_start_x,
            ball_state_after_fire.ball_hit_start_y,
            ball_state_after_fire.ball_hit_target_x,
            ball_state_after_fire.ball_hit_target_y,
            ball_state_after_fire.move_x,
            ball_state_after_fire.move_y,
            ball_state_after_fire.bounces,
            new_last_hit,
        ),
        GameState(
            new_is_serving,
            state.game_state.pause_counter,
            state.game_state.player_score,
            state.game_state.enemy_score,
            state.game_state.player_game_score,
            state.game_state.enemy_game_score,
            state.game_state.is_finished
        ),
        state.counter,
        animator_state=state.animator_state,
        random_key=new_random_key
    )


@jax.jit
def is_overlapping(entity1_x, entity1_w, entity1_y, entity1_h, entity2_x, entity2_w, entity2_y,
                   entity2_h) -> chex.Array:
    """
    Checks if two entities are overlapping, given their positions and size.

    Args:
        entity1_x: The x coordinate of the first entity.
        entity1_w: The width of the first entity.
        entity1_y: The y coordinate of the first entity.
        entity1_h: The height of the first entity.
        entity2_x: The x coordinate of the second entity.
        entity2_w: The width of the second entity.
        entity2_y: The y coordinate of the second entity.
        entity2_h: The height of the second entity.

    Returns:
        chex.Array: True/False depending on whether the two entities are overlapping.
    """

    entity1_end_x = entity1_x + entity1_w
    entity2_end_x = entity2_x + entity2_w
    is_overlapping_x = jnp.logical_not(
        jnp.logical_or(
            entity1_end_x <= entity2_x,
            entity2_end_x <= entity1_x
        )
    )

    entity1_end_y = entity1_y + entity1_h
    entity2_end_y = entity2_y + entity2_h
    is_overlapping_y = jnp.logical_not(
        jnp.logical_or(
            entity1_end_y <= entity2_y,
            entity2_end_y <= entity1_y
        )
    )

    return jnp.logical_and(is_overlapping_x, is_overlapping_y)


@jax.jit
def handle_ball_fire(state: TennisState, hitting_entity_x, hitting_entity_y, direction, consts) -> BallState:
    """
    Adds velocity to the ball based on the hitting entity position.

    Args:
        state (TennisState): Current state of the game.
        hitting_entity_x: X coordinate of the hitting entity.
        hitting_entity_y: Y coordinate of the hitting entity.
        direction: Direction of the hitting entity.

    Returns:
        BallState: New ball state after ball was hit.
    """

    # direction = 1 from top side to bottom
    # direction = -1 from bottom side to top
    # direction = 0 (dont do this)
    new_ball_hit_start_x = state.ball_state.ball_x
    new_ball_hit_start_y = state.ball_state.ball_y

    # hardcoded values could be replaced with constants, but won't make a difference since they shouldn't be changed anyway
    ball_width = 2.0
    max_dist = consts.PLAYER_WIDTH / 2 + ball_width / 2

    angle = -1 * (((hitting_entity_x + consts.PLAYER_WIDTH / 2) - (state.ball_state.ball_x + 2 / 2)) / max_dist) * direction

    hitting_entity_field = jnp.where(
        hitting_entity_y < consts.GAME_MIDDLE,
        1,
        -1
    )
    angle *= hitting_entity_field  # adjust angle in case entity is in bottom field (* -1 will reverse angle)
    # calc x landing position depending on player hit angle
    # angle = 0 is neutral angle, between -1...1
    left_offset = -39
    right_offset = 39
    offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

    y_dist = jnp.where(
        direction > 0,
        # hitting from top to bottom
        jnp.where(
            hitting_entity_y > consts.LONG_HIT_THRESHOLD_TOP,
            consts.SHORT_HIT_DISTANCE,
            consts.LONG_HIT_DISTANCE
        ),
        # hitting from bottom to top
        jnp.where(
            hitting_entity_y < consts.LONG_HIT_THRESHOLD_BOTTOM,
            consts.SHORT_HIT_DISTANCE,
            consts.LONG_HIT_DISTANCE
        )
    )

    new_ball_hit_target_y = new_ball_hit_start_y + (y_dist * direction)
    field_min_x = 32
    field_max_x = 32 + consts.FIELD_WIDTH_TOP
    new_ball_hit_target_x = jnp.clip(new_ball_hit_start_x + offset, field_min_x, field_max_x)

    @jax.jit
    def compute_ball_landing_frame(z_fp0, v0_fp, gravity_fp):
        """
        Computes the number of frames until the ball hits the ground (z == 0).

        Args:
            z_fp0: Initial z fixed-point position (e.g., 0)
            v0_fp: Initial z fixed-point velocity (e.g., 140)
            gravity_fp: Gravity per frame in fixed-point (e.g., 4)

        Returns:
            jnp.ndarray: Number of frames until ball hits the ground
        """
        A = -gravity_fp
        B = 2 * v0_fp + gravity_fp
        C = 2 * (z_fp0 - 10)  # '10' comes from floor(z_fp / 10) == 0  → z_fp < 10

        discriminant = B ** 2 - 4 * A * C
        sqrt_discriminant = jnp.sqrt(discriminant)

        # Only consider the positive root (when the ball actually hits the ground)
        t = (-B - sqrt_discriminant) / (2 * A)

        # Floor to get the last integer frame before hitting ground
        t_landing = jnp.floor(t).astype(jnp.int32)

        return t_landing

    hit_vel = 24.0

    dx = new_ball_hit_target_x - state.ball_state.ball_x
    dy = new_ball_hit_target_y - state.ball_state.ball_y

    steps = compute_ball_landing_frame(14, 24.0, consts.BALL_GRAVITY_PER_FRAME)

    move_x = dx / steps
    move_y = dy / steps

    return BallState(
        state.ball_state.ball_x,
        state.ball_state.ball_y,
        jnp.array(14.0),
        jnp.array(140.0),
        jnp.array(hit_vel),
        new_ball_hit_start_x,
        new_ball_hit_start_y,
        new_ball_hit_target_x,
        new_ball_hit_target_y,
        move_x,
        move_y,
        jnp.array(0),  # ball has not bounced after last hit
        state.ball_state.last_hit
    )

@jax.jit
def tennis_reset(consts) -> TennisState:
    """
    Provides the initial state for the game. For that purpose, we use the default values assigned in TennisState.

    Returns:
        TennisState: The initial state of the game.
    """
    return TennisState(
        player_state=PlayerState(
            player_x = jnp.array(consts.START_X),
            player_y = jnp.array(consts.PLAYER_START_Y),
            player_direction=jnp.array(consts.PLAYER_START_DIRECTION),
            player_field=jnp.array(consts.PLAYER_START_FIELD),
            player_serving=jnp.array(True),
            player_walk_speed=jnp.array(1.0),
        ),
        enemy_state=EnemyState(
            enemy_x=jnp.array(consts.START_X),
            enemy_y=jnp.array(consts.ENEMY_START_Y),
            prev_walking_direction=jnp.array(0.0),
            enemy_direction=jnp.array(consts.PLAYER_START_DIRECTION),
            y_movement_direction=jnp.array(1),
        ),
        ball_state=BallState(
            ball_x=jnp.array(consts.FRAME_WIDTH / 2.0 - consts.BALL_WIDTH / 2),
            ball_y=jnp.array(consts.GAME_OFFSET_TOP),
            ball_z=jnp.array(0.0),
            ball_z_fp=jnp.array(0.0),
            ball_velocity_z_fp=jnp.array(0.0),
            ball_hit_start_x=jnp.array(0.0),
            ball_hit_start_y=jnp.array(0.0),
            ball_hit_target_x=jnp.array(consts.FRAME_WIDTH / 2.0 - consts.BALL_WIDTH / 2),
            ball_hit_target_y=jnp.array(consts.GAME_OFFSET_TOP),
            move_x=jnp.array(0.0),
            move_y=jnp.array(0.0),
            bounces=jnp.array(0),
            last_hit=jnp.array(-1)
        ),
        game_state=GameState(
            is_serving=jnp.array(True),
            pause_counter=jnp.array(0),
            player_score=jnp.array(0),
            enemy_score=jnp.array(0),
            player_game_score=jnp.array(0),
            enemy_game_score=jnp.array(0),
            is_finished=jnp.array(False),
        ),
        counter=jnp.array(0),
        animator_state = AnimatorState(),
        random_key = jax.random.PRNGKey(0)
    )


class PlayerObs(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_field: chex.Array  # top (1) or bottom (-1) field
    player_serving: chex.Array  # true: Player is serving, false: Enemy is serving


class EnemyObs(NamedTuple):
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_direction: chex.Array


class TennisJaxEnv(JaxEnvironment[TennisState, TennisObs, TennisInfo, TennisConstants]):
    # ALE minimal action set: [NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE, UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE]
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
    ], dtype=jnp.int32)

    def __init__(self, consts: TennisConstants = None):
        if consts is None:
            consts = TennisConstants()
        super().__init__(consts)
        self.renderer = TennisRenderer(consts)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[TennisObs, TennisState]:
        """
        Provides the initial state for the game. For that purpose, we use the default values assigned in TennisState.

        Returns:
            TennisState: The initial state of the game.
        """
        reset_state = TennisState(
            player_state=PlayerState(
                player_x = jnp.array(self.consts.START_X),
                player_y = jnp.array(self.consts.PLAYER_START_Y),
                player_direction=jnp.array(self.consts.PLAYER_START_DIRECTION),
                player_field=jnp.array(self.consts.PLAYER_START_FIELD),
                player_serving=jnp.array(True),
                player_walk_speed=jnp.array(1.0),
            ),
            enemy_state=EnemyState(
                enemy_x=jnp.array(self.consts.START_X),
                enemy_y=jnp.array(self.consts.ENEMY_START_Y),
                prev_walking_direction=jnp.array(0.0),
                enemy_direction=jnp.array(self.consts.PLAYER_START_DIRECTION),
                y_movement_direction=jnp.array(1),
            ),
            ball_state=BallState(
                ball_x=jnp.array(self.consts.FRAME_WIDTH / 2.0 - self.consts.BALL_WIDTH / 2),
                ball_y=jnp.array(self.consts.GAME_OFFSET_TOP),
                ball_z=jnp.array(0.0),
                ball_z_fp=jnp.array(0.0),
                ball_velocity_z_fp=jnp.array(0.0),
                ball_hit_start_x=jnp.array(0.0),
                ball_hit_start_y=jnp.array(0.0),
                ball_hit_target_x=jnp.array(self.consts.FRAME_WIDTH / 2.0 - self.consts.BALL_WIDTH / 2),
                ball_hit_target_y=jnp.array(self.consts.GAME_OFFSET_TOP),
                move_x=jnp.array(0.0),
                move_y=jnp.array(0.0),
                bounces=jnp.array(0),
                last_hit=jnp.array(-1)
            ),
            game_state=GameState(
                is_serving=jnp.array(True),
                pause_counter=jnp.array(0),
                player_score=jnp.array(0),
                enemy_score=jnp.array(0),
                player_game_score=jnp.array(0),
                enemy_game_score=jnp.array(0),
                is_finished=jnp.array(False),
            ),
            counter=jnp.array(0),
            animator_state = AnimatorState(),
            random_key = jax.random.PRNGKey(0)
        )
        reset_obs = self._get_observation(reset_state)

        return reset_obs, reset_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TennisState, action) -> Tuple[TennisObs, TennisState, float, bool, TennisInfo]:
        """
        Updates the entire state of the game by calling all step functions.

        Args:
            state (TennisState): The current state of the game.
            action: The action to apply.

        Returns:
            TennisState: The updated state of the game.
        """
        # Translate agent action (0,1,2,...,17) to ALE action
        atari_action = jnp.take(self.ACTION_SET, action)
        
        def _pause_step(state: TennisState) -> TennisState:
            return TennisState(state.player_state, state.enemy_state,
                                state.ball_state,
                                GameState(
                                    state.game_state.is_serving,
                                    state.game_state.pause_counter - 1,
                                    state.game_state.player_score,
                                    state.game_state.enemy_score,
                                    state.game_state.player_game_score,
                                    state.game_state.enemy_game_score,
                                    state.game_state.is_finished,
                                ),
                                state.counter + 1, animator_state=AnimatorState(),random_key=state.random_key)
        
        new_state = jax.lax.cond(state.game_state.is_finished,
                        lambda s: s,
                        lambda s: jax.lax.cond(s.game_state.pause_counter > 0,
                                               _pause_step,
                                               lambda s_inner: self._normal_step(s_inner, atari_action),
                                               s
                                               ),
                        state
                        )
        new_obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _normal_step(self, state: TennisState, atari_action) -> TennisState:
        """
        Updates the entire state of the game by calling all step functions. Should be used when game is not paused.

        Args:
            state (TennisState): The current state of the game.
            atari_action: The translated ALE action to apply.

        Returns:
            TennisState: The updated state of the game.
        """
        new_state_after_score_check = self._check_score(state)
        new_state_after_ball_step = self._ball_step(new_state_after_score_check, atari_action)
        new_player_state = self._player_step(new_state_after_ball_step, atari_action)
        new_enemy_state = self._enemy_step(new_state_after_ball_step)
        new_animator_state = self._animator_step(state, new_player_state, new_enemy_state)

        return TennisState(new_player_state, new_enemy_state, new_state_after_ball_step.ball_state,
                           new_state_after_ball_step.game_state, state.counter + 1, new_animator_state, state.random_key)

    def render(self, state: TennisState) -> Tuple[jnp.ndarray]:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def flatten_player_obs(self, player_obs: PlayerObs):
        return jnp.concatenate([jnp.array([player_obs.player_x.astype(jnp.float64)]), jnp.array([player_obs.player_y.astype(jnp.float64)]),
                                jnp.array([player_obs.player_direction]), jnp.array([player_obs.player_field]),
                                jnp.array([player_obs.player_serving])])

    def flatten_enemy_obs(self, enemy_obs: EnemyObs):
        return jnp.concatenate(
            [jnp.array([enemy_obs.enemy_x.astype(jnp.float64)]), jnp.array([enemy_obs.enemy_y.astype(jnp.float64)]), jnp.array([enemy_obs.enemy_direction])])

    def flatten_ball_obs(self, ball_obs: BallObs):
        return jnp.concatenate(
            [jnp.array([ball_obs.ball_x.astype(jnp.float64)]), jnp.array([ball_obs.ball_y.astype(jnp.float64)]), jnp.array([ball_obs.ball_z.astype(jnp.float64)]),
             jnp.array([ball_obs.bounces]), jnp.array([ball_obs.last_hit])])

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "player_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                "player_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                "player_direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.float32),
                "player_field": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.float32),
                "player_serving": spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
            }),
            "enemy": spaces.Dict({
                "enemy_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                "enemy_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                "enemy_direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.float32),
            }),
            "ball": spaces.Dict({
                "ball_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                "ball_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                "ball_z": spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),
                # no theoretical upper limit, but usually won't go above 50
                "bounces": spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),
                # no theoretical upper limit, but usually won't go above 2 since game is restarted at that point
                "last_hit": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.float32),
            }),
            "is_serving_state": spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
            "player_points": spaces.Box(low=0, high=45, shape=(), dtype=jnp.float32),
            "player_sets": spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),  # no theoretical upper limit
            "enemy_points": spaces.Box(low=0, high=45, shape=(), dtype=jnp.float32),
            "enemy_sets": spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),  # no theoretical upper limit
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.FRAME_HEIGHT, self.consts.FRAME_WIDTH, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_obs(obs.player),
            self.flatten_enemy_obs(obs.enemy),
            self.flatten_ball_obs(obs.ball),
            obs.is_serving_state.flatten(),
            obs.player_points.flatten(),
            obs.player_sets.flatten(),
            obs.enemy_points.flatten(),
            obs.enemy_sets.flatten()
        ])

#jnp.array(x, dtype=jnp.float32)

    def _get_observation(self, state: TennisState) -> TennisObs:
        return TennisObs(player=PlayerObs(state.player_state.player_x.astype(jnp.float32), state.player_state.player_y.astype(jnp.float32),
                                          state.player_state.player_direction.astype(jnp.float32),
                                          state.player_state.player_field.astype(jnp.float32),
                                          jnp.where(state.player_state.player_serving, 1, 0).astype(jnp.float32)),
                         enemy=EnemyObs(state.enemy_state.enemy_x.astype(jnp.float32), state.enemy_state.enemy_y.astype(jnp.float32),
                                        state.enemy_state.enemy_direction.astype(jnp.float32)),
                         ball=BallObs(state.ball_state.ball_x.astype(jnp.float32), state.ball_state.ball_y.astype(jnp.float32), state.ball_state.ball_z.astype(jnp.float32),
                                      state.ball_state.bounces.astype(jnp.float32), state.ball_state.last_hit.astype(jnp.float32)),
                         is_serving_state=jnp.where(state.game_state.is_serving, 1, 0).astype(jnp.float32),
                         player_points=state.game_state.player_score.astype(jnp.float32),
                         enemy_points=state.game_state.enemy_score.astype(jnp.float32),
                         player_sets=state.game_state.player_game_score.astype(jnp.float32),
                         enemy_sets=state.game_state.enemy_game_score.astype(jnp.float32))

        #print(obs)
        # 140564625680144, 140564617071968
        # 140211275917888, 140211243107184
        #return obs

    def _get_info(self, state: TennisState) -> TennisInfo:
        return TennisInfo()

    @partial(jax.jit, static_argnums=(0,))
    def _animator_step(self, state: TennisState, new_player_state, new_enemy_state) -> AnimatorState:
        """
        Updates the animator state depending on the current player and enemy state.

        Args:
            state (TennisState): The current state of the game.
            new_player_state (PlayerState): The new player state, recorded 1 frame after `state`.
            new_enemy_state (EnemyState): The enew enemy state, recorded 1 frame after `state`.

        Returns:
            AnimatorState: The updated animator state.
        """

        @jax.jit
        def check_has_moved(prev_x, cur_x, prev_y, cur_y):
            return jnp.logical_or(
                prev_x != cur_x,
                prev_y != cur_y,
            )

        # Animations for player
        has_player_moved = check_has_moved(state.player_state.player_x, new_player_state.player_x,
                                           state.player_state.player_y, new_player_state.player_y)
        new_player_frame = jnp.where(
            # only update if player has moved, otherwise reset to frame 0
            has_player_moved,
            jnp.where(
                # only update every 4 ticks
                state.counter % 4 == 0,
                state.animator_state.player_frame + 1,
                state.animator_state.player_frame,
            ),
            0
        )

        # if ball is at x/y coordinates of hit location, it was hit in this frame
        was_ball_hit = jnp.logical_and(
            state.ball_state.ball_x == state.ball_state.ball_hit_start_x,
            state.ball_state.ball_y == state.ball_state.ball_hit_start_y,
        )
        player_hit = jnp.logical_or(
            jnp.logical_and(
                state.ball_state.ball_y < self.consts.GAME_MIDDLE,
                state.player_state.player_field == 1
            ),
            jnp.logical_and(
                state.ball_state.ball_y > self.consts.GAME_MIDDLE,
                state.player_state.player_field == -1
            )
        )

        # set player_racket_animation to True if ball was hit
        new_player_racket_animation = jnp.logical_or(
            jnp.logical_and(
                was_ball_hit,
                player_hit
            ),
            state.animator_state.player_racket_animation,
        )
        new_enemy_racket_animation = jnp.logical_or(
            jnp.logical_and(
                was_ball_hit,
                jnp.logical_not(player_hit)
            ),
            state.animator_state.enemy_racket_animation,
        )

        new_player_racket_frame = jnp.where(
            jnp.logical_and(
                new_player_racket_animation,
                state.counter % 4 == 0,
            ),
            (state.animator_state.player_racket_frame + 1) % 4,
            state.animator_state.player_racket_frame,
        )

        new_enemy_racket_frame = jnp.where(
            jnp.logical_and(
                new_enemy_racket_animation,
                state.counter % 4 == 0,
            ),
            (state.animator_state.enemy_racket_frame + 1) % 4,
            state.animator_state.enemy_racket_frame,
        )

        # if animation is over (has reached start frame again), set player_racket_animation to False
        new_player_racket_animation = jnp.where(
            jnp.logical_and(
                new_player_racket_frame == 0,
                state.counter % 4 == 0,
            ),
            False,
            new_player_racket_animation,
        )

        new_enemy_racket_animation = jnp.where(
            jnp.logical_and(
                new_enemy_racket_frame == 0,
                state.counter % 4 == 0,
            ),
            False,
            new_enemy_racket_animation,
        )

        # Animations for enemy
        has_enemy_moved = check_has_moved(state.enemy_state.enemy_x, new_enemy_state.enemy_x, state.enemy_state.enemy_y,
                                          new_enemy_state.enemy_y)
        new_enemy_frame = jnp.where(
            # only update if enemy has moved, otherwise reset to frame 0
            has_enemy_moved,
            jnp.where(
                # only update every 4 ticks
                state.counter % 4 == 0,
                state.animator_state.enemy_frame + 1,
                state.animator_state.enemy_frame,
            ),
            0
        )

        new_animator_state = AnimatorState(new_player_frame % 4, new_enemy_frame % 4, new_player_racket_frame,
                                           new_player_racket_animation, new_enemy_racket_frame, new_enemy_racket_animation)

        return new_animator_state

    @partial(jax.jit, static_argnums=(0,))
    def _check_score(self, state: TennisState) -> TennisState:
        """
        Checks whether a point was scored and updates the state accordingly. Handles re-spawning, changing sides and all necessary restart behavior.

        Args:
            state (TennisState): The current state of the game.

        Returns:
            TennisState: The updated state of the game.
        """

        new_bounces = jnp.where(
            jnp.logical_and(state.ball_state.ball_z <= 0, jnp.logical_not(state.game_state.is_serving)),
            # ball is at z=0 and not serving
            state.ball_state.bounces + 1,
            state.ball_state.bounces
        )

        # update the scores and start pause of game
        after_score_update_game_state = jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(  # If player is top field and ball is bottom field the player scores
                    state.ball_state.ball_y >= self.consts.GAME_MIDDLE,
                    state.player_state.player_field == 1
                ),
                jnp.logical_and(  # If player is bottom field and ball is top field the player scores
                    state.ball_state.ball_y <= self.consts.GAME_MIDDLE,
                    state.player_state.player_field == -1
                )
            ),
            lambda _: GameState(jnp.array(True), jnp.array(self.consts.PAUSE_DURATION), state.game_state.player_score + 1,
                                state.game_state.enemy_score, state.game_state.player_game_score,
                                state.game_state.enemy_game_score, state.game_state.is_finished),
            lambda _: GameState(jnp.array(True), jnp.array(self.consts.PAUSE_DURATION), state.game_state.player_score,
                                state.game_state.enemy_score + 1, state.game_state.player_game_score,
                                state.game_state.enemy_game_score, state.game_state.is_finished),
            None
        )

        # Check if a set has ended and if the game has ended
        after_set_check_game_state = self._check_end(self._check_set(after_score_update_game_state))
        game_point_scored = jnp.logical_or(
            after_set_check_game_state.player_game_score != after_score_update_game_state.player_game_score,
            after_set_check_game_state.enemy_game_score != after_score_update_game_state.enemy_game_score
        )
        should_change_sides = jnp.logical_and(
            game_point_scored,
            (after_set_check_game_state.player_game_score + after_set_check_game_state.enemy_game_score) % 2 != 0
        )

        new_player_field = jnp.where(
            should_change_sides,
            jnp.where(
                state.player_state.player_field == 1,
                -1,
                1
            ),
            state.player_state.player_field
        )

        new_player_serving = jnp.where(
            game_point_scored,
            jnp.where(
                state.player_state.player_serving == True,
                False,
                True
            ),
            state.player_state.player_serving
        )

        ball_spawn_y = jnp.where(
            new_player_serving == True,
            jnp.where(
                new_player_field == 1,
                self.consts.GAME_OFFSET_TOP,
                self.consts.GAME_OFFSET_TOP + self.consts.GAME_HEIGHT
            ),
            jnp.where(
                new_player_field == 1,
                self.consts.GAME_OFFSET_TOP + self.consts.GAME_HEIGHT,
                self.consts.GAME_OFFSET_TOP
            )
        )

        return jax.lax.cond(
            new_bounces >= 2,
            lambda _: TennisState(
                PlayerState(
                    jnp.array(self.consts.START_X),
                    jnp.where(new_player_field == 1, jnp.array(self.consts.PLAYER_START_Y),
                              jnp.array(self.consts.ENEMY_START_Y)),
                    jnp.array(self.consts.PLAYER_START_DIRECTION),
                    new_player_field,
                    new_player_serving,
                    state.player_state.player_walk_speed
                ),
                EnemyState(
                    jnp.array(self.consts.START_X),
                    jnp.where(new_player_field == 1, jnp.array(self.consts.ENEMY_START_Y),
                              jnp.array(self.consts.PLAYER_START_Y)),
                    jnp.array(0.0),
                    jnp.array(self.consts.PLAYER_START_DIRECTION),  # could create ENEMY_START_DIRECTION
                    jnp.array(1)
                ),
                BallState(
                    jnp.array(self.consts.FRAME_WIDTH / 2.0 - self.consts.BALL_WIDTH / 2),
                    ball_spawn_y,
                    jnp.array(0.0),
                    jnp.array(0.0),
                    jnp.array(0.0),
                    jnp.array(0.0),
                    jnp.array(0.0),
                    jnp.array(self.consts.FRAME_WIDTH / 2.0 - self.consts.BALL_WIDTH / 2),
                    jnp.array(self.consts.GAME_OFFSET_TOP),
                    jnp.array(0.0),
                    jnp.array(0.0),
                    jnp.array(0),
                    jnp.array(-1)
                ),
                after_set_check_game_state,
                state.counter,
                animator_state=AnimatorState(),
                random_key=state.random_key,
            ),
            lambda _: TennisState(state.player_state, state.enemy_state, BallState(  # no one has scored yet
                state.ball_state.ball_x,
                state.ball_state.ball_y,
                state.ball_state.ball_z,
                state.ball_state.ball_z_fp,
                state.ball_state.ball_velocity_z_fp,
                state.ball_state.ball_hit_start_x,
                state.ball_state.ball_hit_start_y,
                state.ball_state.ball_hit_target_x,
                state.ball_state.ball_hit_target_y,
                state.ball_state.move_x,
                state.ball_state.move_y,
                new_bounces,
                state.ball_state.last_hit
            ), state.game_state, state.counter, animator_state=state.animator_state,random_key=state.random_key),
            None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_set(self, state: GameState) -> GameState:
        """
        Checks whether the current set has ended and updates the score accordingly.

        Args:
            state (GameState): The current state of the game.

        Returns:
            GameState: The updated state of the game.
        """
        player_won_set = jnp.logical_and(state.player_score >= 4, state.player_score >= state.enemy_score + 2)
        enemy_won_set = jnp.logical_and(state.enemy_score >= 4, state.enemy_score >= state.player_score + 2)

        return jax.lax.cond(
            # Check if set has ended
            jnp.logical_or(player_won_set, enemy_won_set),
            # Set has ended
            lambda _: jax.lax.cond(
                player_won_set,
                # Player has won set
                lambda _: GameState(state.is_serving, state.pause_counter, jnp.array(0), jnp.array(0),
                                    state.player_game_score + 1, state.enemy_game_score, state.is_finished),
                # Enemy has won set
                lambda _: GameState(state.is_serving, state.pause_counter, jnp.array(0), jnp.array(0),
                                    state.player_game_score, state.enemy_game_score + 1, state.is_finished),
                None
            ),
            # Set is still ongoing
            lambda _: state,
            None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_end(self, state: GameState) -> GameState:
        """
        Checks whether the entire game is over and updates the score accordingly.

        Args:
            state (GameState): The current state of the game.

        Returns:
            GameState: The updated state of the game.
        """

        player_won = jnp.logical_and(state.player_game_score >= 6, state.player_game_score >= state.enemy_game_score + 2)
        enemy_won = jnp.logical_and(state.enemy_game_score >= 6, state.enemy_game_score >= state.player_game_score + 2)
        is_finished = jnp.where(jnp.logical_or(player_won, enemy_won), True, False)
        return GameState(state.is_serving, state.pause_counter, state.player_score, state.enemy_score,
                         state.player_game_score, state.enemy_game_score, is_finished)

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: TennisState, atari_action: chex.Array) -> PlayerState:
        """
        Updates player position based on provided action and applies bounding box.

        Args:
            state (PlayerState): The current player state.
            atari_action (chex.Array): The translated ALE action to apply.

        Returns:
            PlayerState: The updated player state.
        """

        player_state = state.player_state

        # does the action contain UP
        up = jnp.any(
            jnp.array(
                [atari_action == Action.UP, atari_action == Action.UPRIGHT, atari_action == Action.UPLEFT,
                 atari_action == Action.UPFIRE, atari_action == Action.UPRIGHTFIRE,
                 atari_action == Action.UPLEFTFIRE]
            )
        )
        # does the action contain DOWN
        down = jnp.any(
            jnp.array(
                [atari_action == Action.DOWN, atari_action == Action.DOWNRIGHT, atari_action == Action.DOWNLEFT,
                 atari_action == Action.DOWNFIRE, atari_action == Action.DOWNRIGHTFIRE,
                 atari_action == Action.DOWNLEFTFIRE]
            )
        )
        # does the action contain LEFT
        left = jnp.any(
            jnp.array(
                [atari_action == Action.LEFT, atari_action == Action.UPLEFT, atari_action == Action.DOWNLEFT,
                 atari_action == Action.LEFTFIRE, atari_action == Action.UPLEFTFIRE,
                 atari_action == Action.DOWNLEFTFIRE]
            )
        )
        # does the action contain RIGHT
        right = jnp.any(
            jnp.array(
                [atari_action == Action.RIGHT, atari_action == Action.UPRIGHT, atari_action == Action.DOWNRIGHT,
                 atari_action == Action.RIGHTFIRE, atari_action == Action.UPRIGHTFIRE,
                 atari_action == Action.DOWNRIGHTFIRE]
            )
        )

        # move left if the player is trying to move left
        player_x = jnp.where(
            left,
            player_state.player_x - player_state.player_walk_speed,
            player_state.player_x,
        )
        # move right if the player is trying to move right
        player_x = jnp.where(
            right,
            player_state.player_x + player_state.player_walk_speed,
            player_x,
        )
        # apply X bounding box
        player_x = jnp.clip(player_x, self.consts.PLAYER_MIN_X, self.consts.PLAYER_MAX_X)

        # move up if the player is trying to move up
        player_y = jnp.where(
            jnp.logical_and(
                up,
                jnp.logical_not(state.game_state.is_serving)  # not allowed to change y position while someone is serving
            ),
            player_state.player_y - player_state.player_walk_speed,
            player_state.player_y,
        )

        # move down if the player is trying to move down
        player_y = jnp.where(
            jnp.logical_and(
                down,
                jnp.logical_not(state.game_state.is_serving)  # not allowed to change y position while someone is serving
            ),
            player_state.player_y + player_state.player_walk_speed,
            player_y,
        )
        # apply Y bounding box
        player_y = jnp.where(
            player_state.player_field == 1,
            jnp.clip(player_y, self.consts.PLAYER_Y_UPPER_BOUND_TOP, self.consts.PLAYER_Y_LOWER_BOUND_TOP),
            jnp.clip(player_y, self.consts.PLAYER_Y_UPPER_BOUND_BOTTOM, self.consts.PLAYER_Y_LOWER_BOUND_BOTTOM)
        )

        new_player_direction = jnp.where(
            player_state.player_x > state.ball_state.ball_x,
            -1,
            player_state.player_direction
        )

        new_player_direction = jnp.where(
            player_state.player_x < state.ball_state.ball_x,
            1,
            new_player_direction
        )

        return PlayerState(
            player_x,
            player_y,
            new_player_direction,
            player_state.player_field,
            player_state.player_serving,
            player_state.player_walk_speed,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: TennisState) -> EnemyState:
        """
        Updates enemy position by following the enemy strategy explained below.

        Enemy strategy:
            x coordinate:
                - just follow x of the ball
                - keep ball at center of sprite so that sprite does not flip often
            y coordinate:
                - rush towards net after hitting ball
                - stay at net for some time
                - turning point (does not quite line up with player hitting ball, usually slightly earlier)
                - move as far away from net as possible
                - after reaching limit sometimes starts moving towards net again

        Args:
            state (TennisState): The current state of the game.

        Returns:
            EnemyState: The updated enemy state.
        """

        # x-coordinate
        enemy_hit_offset = state.enemy_state.prev_walking_direction * 5 * -1 * 0  # this is always 0
        enemy_x_hit_point = state.enemy_state.enemy_x + self.consts.PLAYER_WIDTH / 2 + enemy_hit_offset

        player_x_hit_point = state.player_state.player_x + self.consts.PLAYER_WIDTH / 2
        ball_tracking_tolerance = 1
        x_tracking_tolerance = 2

        # move to middle
        def move_x_to_middle():
            middle_step_x = jnp.where(jnp.less_equal(state.enemy_state.enemy_x, self.consts.GAME_MIDDLE_HORIZONTAL),
                                      state.enemy_state.enemy_x + 1,
                                      state.enemy_state.enemy_x - 1)
            return jnp.where(jnp.abs(state.enemy_state.enemy_x - self.consts.GAME_MIDDLE_HORIZONTAL) > 1, middle_step_x,
                             state.enemy_state.enemy_x)

        def track_ball_x():
            enemy_aiming_x_offset = jnp.where(
                player_x_hit_point < self.consts.FRAME_WIDTH / 2,
                5,
                -15
            )
            diff = state.ball_state.ball_x - (enemy_x_hit_point + enemy_aiming_x_offset)

            # move right if ball is sufficiently to the right
            new_enemy_x = jnp.where(
                diff > x_tracking_tolerance,
                state.enemy_state.enemy_x + 1,
                state.enemy_state.enemy_x
            )

            # move left if ball is sufficiently to the left
            new_enemy_x = jnp.where(
                diff < -x_tracking_tolerance,
                state.enemy_state.enemy_x - 1,
                new_enemy_x
            )
            return new_enemy_x

        new_enemy_x = jax.lax.cond(state.ball_state.last_hit == 1, move_x_to_middle, track_ball_x)

        cur_walking_direction = jnp.where(
            new_enemy_x - state.enemy_state.enemy_x < 0,
            -1,
            state.enemy_state.prev_walking_direction
        )
        cur_walking_direction = jnp.where(
            new_enemy_x - state.enemy_state.enemy_x > 0,
            1,
            cur_walking_direction
        )

        # jnp.clip(new_enemy_x - state.enemy_state.enemy_x, -1, 1))

        should_perform_direction_change = jnp.logical_or(jnp.logical_or(
            jnp.abs((enemy_x_hit_point) - state.ball_state.ball_x) >= ball_tracking_tolerance,
            # state.enemy_state.prev_walking_direction == cur_walking_direction
            False
        ), state.ball_state.last_hit == 1)

        new_enemy_x = jnp.where(should_perform_direction_change, new_enemy_x, state.enemy_state.enemy_x)

        # y-coordinate

        def enemy_y_step():
            state_after_y = jax.lax.cond(
                state.ball_state.last_hit == 1,
                # last hit was enemy rush net
                lambda _: EnemyState(state.enemy_state.enemy_x,
                                     jnp.where(jnp.logical_and(state.enemy_state.enemy_y != self.consts.PLAYER_Y_LOWER_BOUND_TOP,
                                                               state.enemy_state.enemy_y != self.consts.PLAYER_Y_UPPER_BOUND_BOTTOM),
                                               state.enemy_state.enemy_y - state.player_state.player_field,
                                               state.enemy_state.enemy_y
                                               ), state.enemy_state.prev_walking_direction,
                                     state.enemy_state.enemy_direction, jnp.array(1)),
                # last hit was player move away from net until baseline is hit then move towards ball
                lambda _: EnemyState(state.enemy_state.enemy_x,
                                     jnp.where(state.player_state.player_field == 1,
                                               jnp.clip(state.enemy_state.enemy_y +
                                                              state.player_state.player_field * state.enemy_state.y_movement_direction,
                                                              self.consts.PLAYER_Y_UPPER_BOUND_BOTTOM,
                                                              self.consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                               jnp.clip(state.enemy_state.enemy_y +
                                                              state.player_state.player_field * state.enemy_state.y_movement_direction,
                                                              self.consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                              self.consts.PLAYER_Y_LOWER_BOUND_TOP)),
                                     state.enemy_state.prev_walking_direction,
                                     state.enemy_state.enemy_direction,
                                     jnp.where(jnp.logical_or(state.enemy_state.enemy_y == self.consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                              state.enemy_state.enemy_y == self.consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                               jnp.array(-1),
                                               state.enemy_state.y_movement_direction)), operand=None)

            return jax.lax.cond(
                state.game_state.is_serving,
                lambda _: state.enemy_state,
                lambda _: state_after_y,
                operand=None
            )

        enemy_state_after_y_step = enemy_y_step()

        new_enemy_direction = jnp.where(
            state.enemy_state.enemy_x > state.ball_state.ball_x,
            -1,
            state.enemy_state.enemy_direction
        )

        new_enemy_direction = jnp.where(
            state.enemy_state.enemy_x < state.ball_state.ball_x,
            1,
            new_enemy_direction
        )

        return EnemyState(
            new_enemy_x,
            enemy_state_after_y_step.enemy_y,
            cur_walking_direction,
            new_enemy_direction,
            enemy_state_after_y_step.y_movement_direction
        )

    @partial(jax.jit, static_argnums=(0,))
    def _ball_step(self, state: TennisState, atari_action) -> TennisState:
        """
        Updates ball position by applying velocity and gravity. Also handles player-ball collisions
        and fires the ball if the provided action contains FIRE.

        Args:
            state (TennisState): The current state of the game.
            atari_action (chex.Array): The translated ALE action to apply.

        Returns:
            BallState: The updated ball state.
        """

        @jax.jit
        def get_serving_bounce_velocity() -> int:
            """
            Applies a random offset to the base serving bounce velocity.

            Returns:
                int: The calculated bounce velocity
            """

            return self.consts.BALL_SERVING_BOUNCE_VELOCITY_BASE + random.uniform(state.random_key) * self.consts.BALL_SERVING_BOUNCE_VELOCITY_RANDOM_OFFSET

        ball_state = state.ball_state

        # update the fixed-point z velocity value by applying either upward velocity or gravity
        new_ball_velocity_z_fp = jnp.where(
            ball_state.ball_z == 0,
            get_serving_bounce_velocity(),
            ball_state.ball_velocity_z_fp - self.consts.BALL_GRAVITY_PER_FRAME
        )

        # update the fixed-point z value by applying current velocity (can be positive or negative)
        new_ball_z_fp = ball_state.ball_z_fp + new_ball_velocity_z_fp
        # calculate actual z value using floor division by 10, because fixed-point value has exactly one point
        new_ball_z = new_ball_z_fp // 10

        # apply lower bounding box (500 is effectively no MAX bound)
        new_ball_z = jnp.clip(new_ball_z, 0, 500)
        new_ball_z_fp = jnp.clip(new_ball_z_fp, 0, 500)

        # ball movement in x/y direction is linear, no velocity involved
        new_ball_x = ball_state.ball_x + ball_state.move_x
        new_ball_y = ball_state.ball_y + ball_state.move_y

        player_state = state.player_state
        enemy_state = state.enemy_state

        # check player-ball collisions
        player_end = player_state.player_x + self.consts.PLAYER_WIDTH
        ball_end = ball_state.ball_x + self.consts.BALL_WIDTH
        player_overlap_ball_x = jnp.logical_not(
            jnp.logical_or(
                player_end <= ball_state.ball_x,
                ball_end <= player_state.player_x
            )
        )

        # Correct X positions based on facing direction
        corrected_player_x = jnp.where(
            player_state.player_direction == 1,
            player_state.player_x,
            player_state.player_x - self.consts.PLAYER_WIDTH + 2
        )

        corrected_enemy_x = jnp.where(
            enemy_state.enemy_direction == 1,
            enemy_state.enemy_x,
            enemy_state.enemy_x - self.consts.PLAYER_WIDTH + 2
        )

        upper_entity_x = jnp.where(
            player_state.player_field == 1,
            corrected_player_x,
            corrected_enemy_x
        )
        upper_entity_y = jnp.where(
            player_state.player_field == 1,
            player_state.player_y,
            enemy_state.enemy_y
        )

        lower_entity_x = jnp.where(
            player_state.player_field == -1,
            corrected_player_x,
            corrected_enemy_x
        )
        lower_entity_y = jnp.where(
            player_state.player_field == -1,
            player_state.player_y,
            enemy_state.enemy_y
        )

        upper_entity_overlapping_ball = jnp.logical_and(
            self._is_overlapping(
                upper_entity_x,
                self.consts.PLAYER_WIDTH,
                0,  # this is z pos
                self.consts.PLAYER_HEIGHT,
                ball_state.ball_x,
                self.consts.BALL_WIDTH,
                0,  # the original game ignores height when checking collision
                self.consts.BALL_WIDTH  # could be renamed to to BALL_SIZE because ball is square
            ),
            jnp.absolute(upper_entity_y + self.consts.PLAYER_HEIGHT - ball_state.ball_y) <= 3
        )

        lower_entity_overlapping_ball = jnp.logical_and(
            self._is_overlapping(
                lower_entity_x,
                self.consts.PLAYER_WIDTH,
                0,  # this is z pos
                self.consts.PLAYER_HEIGHT,
                ball_state.ball_x,
                self.consts.BALL_WIDTH,
                0,  # the original game ignores height when checking collision
                self.consts.BALL_WIDTH  # could be renamed to to BALL_SIZE because ball is square
            ),
            jnp.absolute(lower_entity_y + self.consts.PLAYER_HEIGHT - ball_state.ball_y) <= 3
        )

        lower_entity_performed_last_hit = jnp.logical_or(
            jnp.logical_and(
                state.player_state.player_field == -1, state.ball_state.last_hit != self.consts.ENEMY_CONST
            ),
            jnp.logical_and(
                state.player_state.player_field == 1, state.ball_state.last_hit != self.consts.PLAYER_CONST
            )
        )

        upper_entity_performed_last_hit = jnp.logical_or(
            jnp.logical_and(
                state.player_state.player_field == 1, state.ball_state.last_hit != self.consts.ENEMY_CONST
            ),
            jnp.logical_and(
                state.player_state.player_field == -1, state.ball_state.last_hit != self.consts.PLAYER_CONST
            )
        )

        # check if fire is pressed
        fire = jnp.any(jnp.array(
            [atari_action == Action.FIRE, atari_action == Action.LEFTFIRE, atari_action == Action.DOWNLEFTFIRE,
             atari_action == Action.DOWNFIRE,
             atari_action == Action.DOWNRIGHTFIRE, atari_action == Action.RIGHTFIRE,
             atari_action == Action.UPRIGHTFIRE, atari_action == Action.UPFIRE,
             atari_action == Action.UPLEFTFIRE]))

        any_entity_ready_to_fire = jnp.logical_or(
            jnp.logical_and(
                upper_entity_overlapping_ball,
                lower_entity_performed_last_hit
            ),
            jnp.logical_and(
                lower_entity_overlapping_ball,
                upper_entity_performed_last_hit
            )
        )

        should_hit = jnp.logical_and(any_entity_ready_to_fire, jnp.logical_or(
            jnp.logical_not(jnp.logical_and(state.player_state.player_serving, state.game_state.is_serving)),
            fire
        ))
        new_is_serving = jnp.where(should_hit, False, state.game_state.is_serving)

        # no need to check whether the lower entity is actually overlapping because this variable won't be used if it isn't
        ball_fire_direction = jnp.where(
            upper_entity_overlapping_ball,
            1,
            -1
        )
        # no need to check whether the lower entity is actually overlapping because this variable won't be used if it isn't
        hitting_entity_x = jnp.where(
            upper_entity_overlapping_ball,
            upper_entity_x,
            lower_entity_x
        )

        hitting_entity_y = jnp.where(
            upper_entity_overlapping_ball,
            upper_entity_y,
            lower_entity_y
        )

        # record which entity hit the ball most recently
        new_last_hit = jnp.where(should_hit,
                                 # player hit
                                 jnp.where(
                                     jnp.logical_or(
                                         jnp.logical_and(upper_entity_overlapping_ball, player_state.player_field == 1),
                                         jnp.logical_and(lower_entity_overlapping_ball, player_state.player_field == -1)
                                     ),
                                     0,
                                     1
                                 ),
                                 state.ball_state.last_hit
                                 )

        ball_state_after_fire = jax.lax.cond(
            should_hit,
            lambda _: self._handle_ball_fire(state, hitting_entity_x, hitting_entity_y, ball_fire_direction),
            lambda _: BallState(
                new_ball_x,
                new_ball_y,
                new_ball_z,
                new_ball_z_fp,
                new_ball_velocity_z_fp,
                ball_state.ball_hit_start_x,
                ball_state.ball_hit_start_y,
                ball_state.ball_hit_target_x,
                ball_state.ball_hit_target_y,
                ball_state.move_x,
                ball_state.move_y,
                ball_state.bounces,
                ball_state.last_hit,
            ),
            None
        )

        new_random_key, _ = jax.random.split(state.random_key)

        return TennisState(
            player_state,
            enemy_state,
            BallState(
                ball_state_after_fire.ball_x,
                ball_state_after_fire.ball_y,
                ball_state_after_fire.ball_z,
                ball_state_after_fire.ball_z_fp,
                ball_state_after_fire.ball_velocity_z_fp,
                ball_state_after_fire.ball_hit_start_x,
                ball_state_after_fire.ball_hit_start_y,
                ball_state_after_fire.ball_hit_target_x,
                ball_state_after_fire.ball_hit_target_y,
                ball_state_after_fire.move_x,
                ball_state_after_fire.move_y,
                ball_state_after_fire.bounces,
                new_last_hit,
            ),
            GameState(
                new_is_serving,
                state.game_state.pause_counter,
                state.game_state.player_score,
                state.game_state.enemy_score,
                state.game_state.player_game_score,
                state.game_state.enemy_game_score,
                state.game_state.is_finished
            ),
            state.counter,
            animator_state=state.animator_state,
            random_key=new_random_key
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_overlapping(self, entity1_x, entity1_w, entity1_y, entity1_h, entity2_x, entity2_w, entity2_y,
                       entity2_h) -> chex.Array:
        """
        Checks if two entities are overlapping, given their positions and size.

        Args:
            entity1_x: The x coordinate of the first entity.
            entity1_w: The width of the first entity.
            entity1_y: The y coordinate of the first entity.
            entity1_h: The height of the first entity.
            entity2_x: The x coordinate of the second entity.
            entity2_w: The width of the second entity.
            entity2_y: The y coordinate of the second entity.
            entity2_h: The height of the second entity.

        Returns:
            chex.Array: True/False depending on whether the two entities are overlapping.
        """

        entity1_end_x = entity1_x + entity1_w
        entity2_end_x = entity2_x + entity2_w
        is_overlapping_x = jnp.logical_not(
            jnp.logical_or(
                entity1_end_x <= entity2_x,
                entity2_end_x <= entity1_x
            )
        )

        entity1_end_y = entity1_y + entity1_h
        entity2_end_y = entity2_y + entity2_h
        is_overlapping_y = jnp.logical_not(
            jnp.logical_or(
                entity1_end_y <= entity2_y,
                entity2_end_y <= entity1_y
            )
        )

        return jnp.logical_and(is_overlapping_x, is_overlapping_y)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_ball_fire(self, state: TennisState, hitting_entity_x, hitting_entity_y, direction) -> BallState:
        """
        Adds velocity to the ball based on the hitting entity position.

        Args:
            state (TennisState): Current state of the game.
            hitting_entity_x: X coordinate of the hitting entity.
            hitting_entity_y: Y coordinate of the hitting entity.
            direction: Direction of the hitting entity.

        Returns:
            BallState: New ball state after ball was hit.
        """

        # direction = 1 from top side to bottom
        # direction = -1 from bottom side to top
        # direction = 0 (dont do this)
        new_ball_hit_start_x = state.ball_state.ball_x
        new_ball_hit_start_y = state.ball_state.ball_y

        # hardcoded values could be replaced with constants, but won't make a difference since they shouldn't be changed anyway
        ball_width = 2.0
        max_dist = self.consts.PLAYER_WIDTH / 2 + ball_width / 2

        angle = -1 * (((hitting_entity_x + self.consts.PLAYER_WIDTH / 2) - (state.ball_state.ball_x + 2 / 2)) / max_dist) * direction

        hitting_entity_field = jnp.where(
            hitting_entity_y < self.consts.GAME_MIDDLE,
            1,
            -1
        )
        angle *= hitting_entity_field  # adjust angle in case entity is in bottom field (* -1 will reverse angle)
        # calc x landing position depending on player hit angle
        # angle = 0 is neutral angle, between -1...1
        left_offset = -39
        right_offset = 39
        offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

        y_dist = jnp.where(
            direction > 0,
            # hitting from top to bottom
            jnp.where(
                hitting_entity_y > self.consts.LONG_HIT_THRESHOLD_TOP,
                self.consts.SHORT_HIT_DISTANCE,
                self.consts.LONG_HIT_DISTANCE
            ),
            # hitting from bottom to top
            jnp.where(
                hitting_entity_y < self.consts.LONG_HIT_THRESHOLD_BOTTOM,
                self.consts.SHORT_HIT_DISTANCE,
                self.consts.LONG_HIT_DISTANCE
            )
        )

        new_ball_hit_target_y = new_ball_hit_start_y + (y_dist * direction)
        field_min_x = 32
        field_max_x = 32 + self.consts.FIELD_WIDTH_TOP
        new_ball_hit_target_x = jnp.clip(new_ball_hit_start_x + offset, field_min_x, field_max_x)

        @jax.jit
        def compute_ball_landing_frame(z_fp0, v0_fp, gravity_fp):
            """
            Computes the number of frames until the ball hits the ground (z == 0).

            Args:
                z_fp0: Initial z fixed-point position (e.g., 0)
                v0_fp: Initial z fixed-point velocity (e.g., 140)
                gravity_fp: Gravity per frame in fixed-point (e.g., 4)

            Returns:
                jnp.ndarray: Number of frames until ball hits the ground
            """
            A = -gravity_fp
            B = 2 * v0_fp + gravity_fp
            C = 2 * (z_fp0 - 10)  # '10' comes from floor(z_fp / 10) == 0  → z_fp < 10

            discriminant = B ** 2 - 4 * A * C
            sqrt_discriminant = jnp.sqrt(discriminant)

            # Only consider the positive root (when the ball actually hits the ground)
            t = (-B - sqrt_discriminant) / (2 * A)

            # Floor to get the last integer frame before hitting ground
            t_landing = jnp.floor(t).astype(jnp.int32)

            return t_landing

        hit_vel = 24.0

        dx = new_ball_hit_target_x - state.ball_state.ball_x
        dy = new_ball_hit_target_y - state.ball_state.ball_y

        steps = compute_ball_landing_frame(14, 24.0, self.consts.BALL_GRAVITY_PER_FRAME)

        move_x = dx / steps
        move_y = dy / steps

        return BallState(
            state.ball_state.ball_x,
            state.ball_state.ball_y,
            jnp.array(14.0),
            jnp.array(140.0),
            jnp.array(hit_vel),
            new_ball_hit_start_x,
            new_ball_hit_start_y,
            new_ball_hit_target_x,
            new_ball_hit_target_y,
            move_x,
            move_y,
            jnp.array(0),  # ball has not bounced after last hit
            state.ball_state.last_hit
        )

    def _get_reward(self, previous_state: TennisState, state: TennisState) -> float:
        return 0.0

    def _get_done(self, state: TennisState) -> bool:
        return state.game_state.is_finished


class TennisRenderer(JAXGameRenderer):

    def __init__(self, consts: TennisConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or TennisConstants()
        super().__init__(self.consts)
        self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "tennis")
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.FRAME_HEIGHT, self.consts.FRAME_WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 2. Load and pad background
        background_rgba = self.jr.loadFrame(os.path.join(self.sprite_path, 'background.npy'))
        bg_h, bg_w = background_rgba.shape[0], background_rgba.shape[1]
        frame_h, frame_w = self.config.game_dimensions
        if bg_h != frame_h or bg_w != frame_w:
            pad_h_total = frame_h - bg_h
            pad_w_total = frame_w - bg_w
            pad_h_before = pad_h_total // 2
            pad_h_after = pad_h_total - pad_h_before
            pad_w_before = pad_w_total // 2
            pad_w_after = pad_w_total - pad_w_before
            padded_bg = jnp.zeros((frame_h, frame_w, 4), dtype=jnp.uint8)
            padded_bg = padded_bg.at[:, :, 3].set(255)
            padded_bg = padded_bg.at[pad_h_before:pad_h_before+bg_h, pad_w_before:pad_w_before+bg_w, :].set(background_rgba)
            background_rgba = padded_bg
            
        # 3. Get asset manifest
        # Get file-based assets from consts
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # Create procedural assets
        procedural_colors = jnp.array([
            [117, 128, 240, 255],
            [240, 128, 128, 255],
        ], dtype=jnp.uint8).reshape(-1, 1, 1, 4)
        
        # Add procedural and specially-handled background assets
        final_asset_config.append(
            {'name': 'background', 'type': 'background', 'data': background_rgba}
        )
        final_asset_config.append(
            {'name': 'swatch', 'type': 'procedural', 'data': procedural_colors}
        )
        
        # 4. Load assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)
        # 5. Get color IDs
        self.BLUE_ID = self.COLOR_TO_ID.get((117, 128, 240), 0)
        self.RED_ID = self.COLOR_TO_ID.get((240, 128, 128), 0)
        
        if self.BLUE_ID == 0 or self.RED_ID == 0:
            blue_rgb = np.array([117, 128, 240], dtype=np.uint8)
            red_rgb = np.array([240, 128, 128], dtype=np.uint8)
            palette_np = np.array(self.PALETTE)
            palette_rgb = palette_np[:, :3] if palette_np.shape[1] >= 3 else palette_np
            if self.BLUE_ID == 0:
                blue_dists = np.sum(np.abs(palette_rgb.astype(np.int32) - blue_rgb.astype(np.int32)), axis=1)
                self.BLUE_ID = int(np.argmin(blue_dists))
            if self.RED_ID == 0:
                red_dists = np.sum(np.abs(palette_rgb.astype(np.int32) - red_rgb.astype(np.int32)), axis=1)
                self.RED_ID = int(np.argmin(red_dists))
        # 6. Create color-swapped assets
        def swap_colors(mask):
            return jnp.where(mask == self.BLUE_ID, self.RED_ID, 
                   jnp.where(mask == self.RED_ID, self.BLUE_ID, mask))
        vmap_swap = jax.vmap(swap_colors)
        
        self.BG_TOP_MASK = self.BACKGROUND
        self.BG_BOTTOM_MASK = swap_colors(self.BG_TOP_MASK)
        
        self.PLAYER_STACK = self.SHAPE_MASKS['player_anim']
        self.ENEMY_STACK = vmap_swap(self.PLAYER_STACK)
        
        self.PLAYER_RACKET_STACK = self.SHAPE_MASKS['racket_anim']
        self.ENEMY_RACKET_STACK = vmap_swap(self.PLAYER_RACKET_STACK)
        
        self.PLAYER_DIGITS_STACK = self.SHAPE_MASKS['digits']
        self.ENEMY_DIGITS_STACK = vmap_swap(self.PLAYER_DIGITS_STACK)
        
        self.PLAYER_UI_A = self.SHAPE_MASKS['ui_a']
        self.ENEMY_UI_A = swap_colors(self.PLAYER_UI_A)
        # 7. Store animation lengths
        self.anim_len = {
            'player': self.ENEMY_STACK.shape[0],
            'racket': self.ENEMY_RACKET_STACK.shape[0],
        }

    @partial(jax.jit, static_argnums=(0,))
    def _render_score(self, raster, state: TennisState, bg_top_red: bool) -> jnp.ndarray:
        digit_stack, ui_a_mask = jax.lax.cond(
            bg_top_red,
            lambda: (self.ENEMY_DIGITS_STACK, self.ENEMY_UI_A),
            lambda: (self.PLAYER_DIGITS_STACK, self.PLAYER_UI_A)
        )
        
        should_display_overall_score = (
            (state.game_state.player_game_score + state.game_state.enemy_game_score) > 0
        ) & (
            (state.game_state.player_score == 0) & (state.game_state.enemy_score == 0)
        )
        def render_overall(r):
            player_digit_stack = self.PLAYER_DIGITS_STACK
            enemy_digit_stack = self.ENEMY_DIGITS_STACK
            r = self._render_number_centered(r, state.game_state.player_game_score, 
                                             (self.consts.FRAME_WIDTH // 4) * 3, 2, 
                                             player_digit_stack)
            r = self._render_number_centered(r, state.game_state.enemy_game_score,
                                             self.consts.FRAME_WIDTH // 4, 2,
                                             enemy_digit_stack)
            return r
        def render_current_score(r):
            tennis_scores = jnp.array([0, 15, 30, 40], dtype=jnp.int32)
            ps = state.game_state.player_score
            es = state.game_state.enemy_score
            serving = state.player_state.player_serving
            deuce_like = (ps >= 3) & (es >= 3)
            def render_deuce(r_in):
                is_tied = (ps == es)
                ad_in = ((ps > es) & serving) | ((es > ps) & (~serving))
                ui_mask = jax.lax.select(ad_in, self.PLAYER_UI_A, self.ENEMY_UI_A)
                x_pos = (self.consts.FRAME_WIDTH // 4) - (ui_mask.shape[1] // 2)
                r_out = jax.lax.cond(
                    is_tied,
                    lambda r_l: self.jr.render_at(r_l, x_pos, 2, self.ENEMY_UI_A, flip_offset=self.FLIP_OFFSETS['ui_a']),
                    lambda r_l: self.jr.render_at(r_l, x_pos, 2, ui_mask, flip_offset=self.FLIP_OFFSETS['ui_a']),
                    r_in
                )
                return r_out
            def render_regular(r_in):
                pid = jnp.minimum(3, ps)
                eid = jnp.minimum(3, es)
                pnum = tennis_scores[pid]
                enum = tennis_scores[eid]
                player_digit_stack = self.PLAYER_DIGITS_STACK
                enemy_digit_stack = self.ENEMY_DIGITS_STACK
                r_out = self._render_number_centered(r_in, pnum, (self.consts.FRAME_WIDTH // 4) * 3, 2, player_digit_stack)
                r_out = self._render_number_centered(r_out, enum, self.consts.FRAME_WIDTH // 4, 2, enemy_digit_stack)
                return r_out
            return jax.lax.cond(deuce_like, render_deuce, render_regular, r)
        return jax.lax.cond(
            should_display_overall_score,
            render_overall,
            render_current_score,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_number_centered(self, raster, number, center_x, y_pos, digit_stack):
        digits = self.jr.int_to_digits(number, max_digits=2)
        n = jnp.asarray(number, jnp.int32)
        is_single = n < 10
        start_idx = jax.lax.select(is_single, 1, 0)
        count = jax.lax.select(is_single, 1, 2)
        
        digit_width = self.SHAPE_MASKS['digits'].shape[2]
        spacing = 7
        total_width = count * digit_width + (count - 1) * (spacing - digit_width)
        x_start = center_x - (total_width // 2)
        return self.jr.render_label_selective(
            raster, x_start, y_pos,
            digits, digit_stack,
            start_idx, count,
            spacing=spacing,
            max_digits_to_render=2
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TennisState) -> jnp.ndarray:
        # 1. Select background mask
        bg_top_red = True
        bg_mask = self.BG_TOP_MASK
        
        raster = self.jr.create_object_raster(bg_mask)
        # 2. Render Ball Shadow
        raster = self.jr.render_at_clipped(
            raster, state.ball_state.ball_x, state.ball_state.ball_y,
            self.SHAPE_MASKS['ball_shadow'], 
            flip_offset=self.FLIP_OFFSETS['ball_shadow']
        )
        
        # 3. Render Player & Enemy
        player_mask_stack = self.PLAYER_STACK
        player_racket_stack = self.PLAYER_RACKET_STACK
        enemy_mask_stack = self.ENEMY_STACK
        enemy_racket_stack = self.ENEMY_RACKET_STACK
        
        player_anim_idx = state.animator_state.player_frame % self.anim_len['player']
        player_racket_idx = state.animator_state.player_racket_frame
        player_mask = player_mask_stack[player_anim_idx]
        player_racket_mask = player_racket_stack[player_racket_idx]
        
        enemy_anim_idx = state.animator_state.enemy_frame % self.anim_len['player']
        enemy_racket_idx = state.animator_state.enemy_racket_frame
        enemy_mask = enemy_mask_stack[enemy_anim_idx]
        enemy_racket_mask = enemy_racket_stack[enemy_racket_idx]

        # --- Player ---
        player_pos_x = jnp.where(state.player_state.player_direction == 1,
                                 state.player_state.player_x - 2,
                                 state.player_state.player_x - 4)
        flip_player = state.player_state.player_direction == -1
        
        raster = self.jr.render_at_clipped(
            raster, player_pos_x, state.player_state.player_y, player_mask,
            flip_horizontal=flip_player, flip_offset=self.FLIP_OFFSETS['player_anim']
        )
        
        racket_offset_x_map = jnp.asarray([0, 1, 2, 2])
        racket_offset_y_map = jnp.asarray([1, 8, 8, 4])
        
        base_hand_x_right = 5
        flip_offset_x = self.FLIP_OFFSETS['racket_anim'][0]
        player_racket_pos_x = jnp.where(
            state.player_state.player_direction == 1,
            player_pos_x + base_hand_x_right + racket_offset_x_map[player_racket_idx],
            player_pos_x - racket_offset_x_map[player_racket_idx] - 5 + flip_offset_x
        )
        player_racket_pos_y = state.player_state.player_y + racket_offset_y_map[player_racket_idx]
        raster = self.jr.render_at_clipped(
            raster, player_racket_pos_x, player_racket_pos_y,
            player_racket_mask,
            flip_horizontal=flip_player, flip_offset=self.FLIP_OFFSETS['racket_anim']
        )

        # --- Enemy ---
        enemy_pos_x = jnp.where(state.enemy_state.enemy_direction == 1,
                                state.enemy_state.enemy_x - 2,
                                state.enemy_state.enemy_x - 4)
        flip_enemy = state.enemy_state.enemy_direction == -1
        
        raster = self.jr.render_at_clipped(
            raster, enemy_pos_x, state.enemy_state.enemy_y, enemy_mask,
            flip_horizontal=flip_enemy, flip_offset=self.FLIP_OFFSETS['player_anim']
        )
        
        base_hand_x_right = 5
        flip_offset_x = self.FLIP_OFFSETS['racket_anim'][0]
        enemy_racket_pos_x = jnp.where(
            state.enemy_state.enemy_direction == 1,
            enemy_pos_x + base_hand_x_right + racket_offset_x_map[enemy_racket_idx],
            enemy_pos_x - racket_offset_x_map[enemy_racket_idx] - 5 + flip_offset_x
        )
        enemy_racket_pos_y = state.enemy_state.enemy_y + racket_offset_y_map[enemy_racket_idx]
        
        raster = self.jr.render_at_clipped(
            raster, enemy_racket_pos_x, enemy_racket_pos_y,
            enemy_racket_mask,
            flip_horizontal=flip_enemy, flip_offset=self.FLIP_OFFSETS['racket_anim']
        )
        
        # 4. Render Ball
        raster = self.jr.render_at_clipped(
            raster, state.ball_state.ball_x, state.ball_state.ball_y - state.ball_state.ball_z,
            self.SHAPE_MASKS['ball'], flip_offset=self.FLIP_OFFSETS['ball']
        )
        
        # 5. Render Score UI
        raster = self._render_score(raster, state, bg_top_red)

        # 6. Final Palette Lookup
        return self.jr.render_from_palette(raster, self.PALETTE)