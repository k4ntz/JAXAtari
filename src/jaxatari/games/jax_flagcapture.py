from functools import partial
import os
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.spaces import Space
from jaxatari.environment import JAXAtariAction as Action


#
# by Tim Morgner and Jan Larionow
#

class FlagCaptureConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    SCALING_FACTOR: int = struct.field(pytree_node=False, default=3)

    @property
    def WINDOW_WIDTH(self) -> int:
        return self.WIDTH * self.SCALING_FACTOR

    @property
    def WINDOW_HEIGHT(self) -> int:
        return self.HEIGHT * self.SCALING_FACTOR

    PLAYER_STATUS_ALIVE: int = struct.field(pytree_node=False, default=0)
    PLAYER_STATUS_BOMB: int = struct.field(pytree_node=False, default=1)
    PLAYER_STATUS_FLAG: int = struct.field(pytree_node=False, default=2)
    PLAYER_STATUS_NUMBER_1: int = struct.field(pytree_node=False, default=3)
    PLAYER_STATUS_NUMBER_2: int = struct.field(pytree_node=False, default=4)
    PLAYER_STATUS_NUMBER_3: int = struct.field(pytree_node=False, default=5)
    PLAYER_STATUS_NUMBER_4: int = struct.field(pytree_node=False, default=6)
    PLAYER_STATUS_NUMBER_5: int = struct.field(pytree_node=False, default=7)
    PLAYER_STATUS_NUMBER_6: int = struct.field(pytree_node=False, default=8)
    PLAYER_STATUS_NUMBER_7: int = struct.field(pytree_node=False, default=9)
    PLAYER_STATUS_NUMBER_8: int = struct.field(pytree_node=False, default=10)
    PLAYER_STATUS_DIRECTION_UP: int = struct.field(pytree_node=False, default=11)
    PLAYER_STATUS_DIRECTION_RIGHT: int = struct.field(pytree_node=False, default=12)
    PLAYER_STATUS_DIRECTION_DOWN: int = struct.field(pytree_node=False, default=13)
    PLAYER_STATUS_DIRECTION_LEFT: int = struct.field(pytree_node=False, default=14)
    PLAYER_STATUS_DIRECTION_UPRIGHT: int = struct.field(pytree_node=False, default=15)
    PLAYER_STATUS_DIRECTION_UPLEFT: int = struct.field(pytree_node=False, default=16)
    PLAYER_STATUS_DIRECTION_DOWNRIGHT: int = struct.field(pytree_node=False, default=17)
    PLAYER_STATUS_DIRECTION_DOWNLEFT: int = struct.field(pytree_node=False, default=18)
    PLAYER_STATUS_CLUE_PLACEHOLDER_NUMBER: int = struct.field(pytree_node=False, default=19)
    PLAYER_STATUS_CLUE_PLACEHOLDER_DIRECTION: int = struct.field(pytree_node=False, default=20)

    NUM_FIELDS_X: int = struct.field(pytree_node=False, default=9)
    NUM_FIELDS_Y: int = struct.field(pytree_node=False, default=7)

    FIELD_PADDING_LEFT: int = struct.field(pytree_node=False, default=12)
    FIELD_PADDING_TOP: int = struct.field(pytree_node=False, default=26)
    FIELD_GAP_X: int = struct.field(pytree_node=False, default=8)
    FIELD_GAP_Y: int = struct.field(pytree_node=False, default=8)
    FIELD_WIDTH: int = struct.field(pytree_node=False, default=8)
    FIELD_HEIGHT: int = struct.field(pytree_node=False, default=16)
    SCORE_AND_TIMER_PADDING_TOP: int = struct.field(pytree_node=False, default=7)

    NUM_BOMBS: int = struct.field(pytree_node=False, default=3)
    NUM_NUMBER_CLUES: int = struct.field(pytree_node=False, default=30)
    NUM_DIRECTION_CLUES: int = struct.field(pytree_node=False, default=30)
    MOVE_COOLDOWN: int = struct.field(pytree_node=False, default=15)
    STEPS_PER_SECOND: int = struct.field(pytree_node=False, default=60)

    ANIMATION_TYPE_NONE: int = struct.field(pytree_node=False, default=0)
    ANIMATION_TYPE_EXPLOSION: int = struct.field(pytree_node=False, default=1)
    ANIMATION_TYPE_FLAG: int = struct.field(pytree_node=False, default=2)

@struct.dataclass
class FlagCaptureState:
    player_x: chex.Array
    player_y: chex.Array
    time: chex.Array
    is_checking: chex.Array
    score: chex.Array
    field: chex.Array
    player_move_cooldown: chex.Array
    animation_cooldown: chex.Array
    animation_type: chex.Array
    rng_key: chex.PRNGKey

@struct.dataclass
class PlayerEntity:
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array
    status: chex.Array


@struct.dataclass
class FlagCaptureObservation:
    player: PlayerEntity
    score: chex.Array


@struct.dataclass
class FlagCaptureInfo:
    time: chex.Array
    score: chex.Array


class JaxFlagCapture(JaxEnvironment[FlagCaptureState, FlagCaptureObservation, FlagCaptureInfo,FlagCaptureConstants]):
    # Minimal ALE action set for Flag Capture
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
    
    def __init__(self,consts:FlagCaptureConstants = None):
        consts = consts or FlagCaptureConstants()
        super().__init__(consts)
        self.renderer = FlagCaptureRenderer(consts=consts)

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(187)) -> Tuple[
        FlagCaptureObservation, FlagCaptureState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)

        Args:
            key: Random key for generating the initial state.
        Returns:
            initial_obs: Initial observation of the game.
            state: Initial game state.
        """
        newkey, splitkey = jax.random.split(key)
        generated_field = self.generate_field(splitkey, self.consts.NUM_BOMBS, self.consts.NUM_NUMBER_CLUES, self.consts.NUM_DIRECTION_CLUES)
        state = FlagCaptureState(
            player_x=jnp.array(0).astype(jnp.int32),
            player_y=jnp.array(0).astype(jnp.int32),
            time=jnp.array(75 * self.consts.STEPS_PER_SECOND).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            is_checking=jnp.array(0).astype(jnp.int32),
            player_move_cooldown=jnp.array(0).astype(jnp.int32),
            animation_cooldown=jnp.array(0).astype(jnp.int32),
            animation_type=jnp.array(0).astype(jnp.int32),
            field=generated_field,
            rng_key=newkey,
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FlagCaptureState):
        """
        Returns the observation of the game state.
        Args:
            state: The current game state.
        Returns:
            FlagCaptureObservation: The observation of the game state.
        """
        return FlagCaptureObservation(
            player=PlayerEntity(
                x=jnp.array(
                    self.consts.FIELD_PADDING_LEFT + (state.player_x * self.consts.FIELD_WIDTH) + (
                                state.player_x * self.consts.FIELD_GAP_X)).astype(
                    jnp.int32),
                y=jnp.array(
                    self.consts.FIELD_PADDING_TOP + (state.player_y * self.consts.FIELD_HEIGHT) + (
                                state.player_y * self.consts.FIELD_GAP_Y)).astype(
                    jnp.int32),
                width=jnp.array(self.consts.FIELD_WIDTH).astype(jnp.int32),
                height=jnp.array(self.consts.FIELD_HEIGHT).astype(jnp.int32),
                status=jnp.array(self.consts.PLAYER_STATUS_ALIVE).astype(jnp.int32),
            ),
            score=state.score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FlagCaptureState, state: FlagCaptureState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FlagCaptureState) -> bool:
        return state.time <= 0

    def action_space(self) -> Space:
        """
        Returns the action space of the environment.
        Returns: The action space of the environment as a Discrete space.
        """
        return spaces.Discrete(18)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "status": spaces.Box(low=0, high=20, shape=(), dtype=jnp.int32),
            }),
            "score": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    def obs_to_flat_array(self, obs: FlagCaptureObservation) -> jnp.ndarray:
        return jnp.array([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.player.status.flatten(),
            obs.score.flatten(),
        ]).flatten()

    def render(self, state: FlagCaptureState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FlagCaptureState, action: chex.Array):
        """
        Takes a step in the game environment based on the action taken.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            observation: The new observation of the game state.
            new_state: The new game state after taking the action.
            reward: The reward received after taking the action.
            done: A boolean indicating if the game is over.
            info: Additional information about the game state.
        """
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        def player_step(player_x, player_y, is_checking, player_move_cooldown, animation_type, action):
            """
            Updates the player's position and state based on the action taken.

            Args:
                player_x: Current x position of the player.
                player_y: Current y position of the player.
                is_checking: Current state of the player (checking or not).
                player_move_cooldown: Current cooldown for player movement.
                animation_type: Current animation type.
                action: Action taken by the player.

            Returns:
                new_player_x: Updated x position of the player.
                new_player_y: Updated y position of the player.
                new_is_checking: Updated state of the player (checking or not).
                new_player_move_cooldown: Updated cooldown for player movement.
            """
            new_is_checking = jax.lax.cond(jnp.logical_or(jnp.equal(action, JAXAtariAction.FIRE),
                                                          jnp.logical_and(
                                                              jnp.greater_equal(action, JAXAtariAction.UPFIRE),
                                                              jnp.less_equal(action, JAXAtariAction.DOWNLEFTFIRE))),
                                           lambda: 1, lambda: 0)
            is_up = jnp.logical_or(jnp.equal(action, JAXAtariAction.UP),
                                   jnp.logical_or(jnp.equal(action, JAXAtariAction.UPFIRE),
                                                  jnp.logical_or(jnp.equal(action, JAXAtariAction.UPRIGHT),
                                                                 jnp.logical_or(
                                                                     jnp.equal(action, JAXAtariAction.UPLEFT),
                                                                     jnp.logical_or(jnp.equal(action,
                                                                                              JAXAtariAction.UPRIGHTFIRE),
                                                                                    jnp.equal(action,
                                                                                              JAXAtariAction.UPLEFTFIRE))))))
            is_down = jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWN),
                                     jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNFIRE),
                                                    jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNRIGHT),
                                                                   jnp.logical_or(
                                                                       jnp.equal(action, JAXAtariAction.DOWNLEFT),
                                                                       jnp.logical_or(jnp.equal(action,
                                                                                                JAXAtariAction.DOWNRIGHTFIRE),
                                                                                      jnp.equal(action,
                                                                                                JAXAtariAction.DOWNLEFTFIRE))))))
            is_left = jnp.logical_or(jnp.equal(action, JAXAtariAction.LEFT),
                                     jnp.logical_or(jnp.equal(action, JAXAtariAction.UPLEFT),
                                                    jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNLEFT),
                                                                   jnp.logical_or(
                                                                       jnp.equal(action, JAXAtariAction.LEFTFIRE),
                                                                       jnp.logical_or(jnp.equal(action,
                                                                                                JAXAtariAction.UPLEFTFIRE),
                                                                                      jnp.equal(action,
                                                                                                JAXAtariAction.DOWNLEFTFIRE))))))
            is_right = jnp.logical_or(jnp.equal(action, JAXAtariAction.RIGHT),
                                      jnp.logical_or(jnp.equal(action, JAXAtariAction.UPRIGHT),
                                                     jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNRIGHT),
                                                                    jnp.logical_or(
                                                                        jnp.equal(action, JAXAtariAction.RIGHTFIRE),
                                                                        jnp.logical_or(jnp.equal(action,
                                                                                                 JAXAtariAction.UPRIGHTFIRE),
                                                                                       jnp.equal(action,
                                                                                                 JAXAtariAction.DOWNRIGHTFIRE))))))

            new_player_y = jax.lax.cond(is_down, lambda: player_y + 1, lambda: player_y)
            new_player_y = jax.lax.cond(is_up, lambda: player_y - 1, lambda: new_player_y)
            new_player_x = jax.lax.cond(is_left, lambda: player_x - 1, lambda: player_x)
            new_player_x = jax.lax.cond(is_right, lambda: player_x + 1, lambda: new_player_x)
            new_player_x = jax.lax.cond(jnp.logical_or(new_is_checking,
                                                       jnp.logical_or(jnp.not_equal(animation_type,
                                                                                    self.consts.ANIMATION_TYPE_NONE),
                                                                      jnp.greater(player_move_cooldown, 0))),
                                        lambda: player_x,
                                        lambda: jnp.mod(new_player_x, self.consts.NUM_FIELDS_X))
            new_player_y = jax.lax.cond(jnp.logical_or(new_is_checking,
                                                       jnp.logical_or(jnp.not_equal(animation_type,
                                                                                    self.consts.ANIMATION_TYPE_NONE),
                                                                      jnp.greater(player_move_cooldown, 0))),
                                        lambda: player_y,
                                        lambda: jnp.mod(new_player_y, self.consts.NUM_FIELDS_Y))
            new_player_move_cooldown = jax.lax.cond(jnp.less_equal(player_move_cooldown, 0),
                                                    lambda: self.consts.MOVE_COOLDOWN,
                                                    lambda: player_move_cooldown - 1)

            return new_player_x, new_player_y, new_is_checking, new_player_move_cooldown

        new_player_x, new_player_y, new_is_checking, new_player_move_cooldown = player_step(
            state.player_x,
            state.player_y,
            state.is_checking,
            state.player_move_cooldown,
            state.animation_type,
            atari_action,
        )

        bomb_animation_over = jnp.logical_and(
            jnp.less_equal(state.animation_cooldown, 0),
            jnp.equal(state.animation_type, self.consts.ANIMATION_TYPE_EXPLOSION)
        )
        flag_animation_over = jnp.logical_and(
            jnp.less_equal(state.animation_cooldown, 0),
            jnp.equal(state.animation_type, self.consts.ANIMATION_TYPE_FLAG)
        )

        new_score = jax.lax.cond(
            flag_animation_over,
            lambda: state.score + 1,
            lambda: state.score
        )
        rng_key, splitkey = jax.random.split(state.rng_key)
        new_field = jax.lax.cond(
            flag_animation_over,
            lambda: self.generate_field(splitkey, self.consts.NUM_BOMBS, self.consts.NUM_NUMBER_CLUES,
                                        self.consts.NUM_DIRECTION_CLUES),
            lambda: state.field
        )

        new_player_x = jax.lax.cond(jnp.logical_or(bomb_animation_over, flag_animation_over),
                                    lambda: 0,
                                    lambda: new_player_x)
        new_player_y = jax.lax.cond(jnp.logical_or(bomb_animation_over, flag_animation_over),
                                    lambda: 0,
                                    lambda: new_player_y)

        new_animation_type = jax.lax.cond(
            jnp.logical_or(bomb_animation_over, flag_animation_over),
            lambda: self.consts.ANIMATION_TYPE_NONE,
            lambda: jax.lax.cond(
                jnp.logical_and(jnp.equal(state.field[new_player_x, new_player_y], self.consts.PLAYER_STATUS_BOMB),
                                jnp.logical_and(
                                    jnp.equal(new_is_checking, 1),
                                    jnp.equal(state.animation_type, self.consts.ANIMATION_TYPE_NONE))),
                lambda: self.consts.ANIMATION_TYPE_EXPLOSION,
                lambda: jax.lax.cond(
                    jnp.logical_and(jnp.equal(state.field[new_player_x, new_player_y], self.consts.PLAYER_STATUS_FLAG),
                                    jnp.logical_and(
                                        jnp.equal(new_is_checking, 1),
                                        jnp.equal(state.animation_type, self.consts.ANIMATION_TYPE_NONE))),
                    lambda: self.consts.ANIMATION_TYPE_FLAG,
                    lambda: state.animation_type))
        )

        new_animation_cooldown = jax.lax.cond(
            jnp.logical_and(jnp.equal(state.animation_type, self.consts.ANIMATION_TYPE_NONE),
                            jnp.equal(new_animation_type, self.consts.ANIMATION_TYPE_EXPLOSION)),
            lambda: 15,
            lambda: jax.lax.cond(
                jnp.logical_and(jnp.equal(state.animation_type, self.consts.ANIMATION_TYPE_NONE),
                                jnp.equal(new_animation_type, self.consts.ANIMATION_TYPE_FLAG)),
                lambda: 30,
                lambda: jax.lax.cond(state.animation_cooldown > 0, lambda: state.animation_cooldown - 1, lambda: 0)))

        new_time = state.time - 1

        new_state = jax.lax.cond(jax.lax.le(state.time, 0),
                                 lambda: FlagCaptureState(
                                     player_x=state.player_x,
                                     player_y=state.player_y,
                                     time=state.time,
                                     is_checking=state.is_checking,
                                     score=state.score,
                                     field=state.field,
                                     player_move_cooldown=state.player_move_cooldown,
                                     animation_cooldown=new_animation_cooldown,
                                     animation_type=state.animation_type,
                                     rng_key=rng_key,
                                 ),
                                 lambda: FlagCaptureState(
                                     player_x=new_player_x,
                                     player_y=new_player_y,
                                     time=new_time,
                                     is_checking=new_is_checking,
                                     score=new_score,
                                     field=new_field,
                                     player_move_cooldown=new_player_move_cooldown,
                                     animation_cooldown=new_animation_cooldown,
                                     animation_type=new_animation_type,
                                     rng_key=rng_key,
                                 ))

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def generate_field(self, rng_key, n_bombs, n_number_clues, n_direction_clues):
        """
        Generates a game field with a flag, bombs, number clues, and direction clues.
        If too few bombs+clues are provided, the rest of the field will be filled with empty spaces.
        (This is not a possibility in the original game, but is there to handle misconfigurations)

        Args:
            field: A 2D array representing the game field.
            n_bombs: Number of bombs to place on the field.
            n_number_clues: Number of number clues to place on the field.
            n_direction_clues: Number of direction clues to place on the field.

        Returns:
            field: A 2D array representing the game field with the placed items.
        """
        rng_key, splitkey = jax.random.split(rng_key)
        flag_x = jax.random.randint(splitkey, (), 0, self.consts.NUM_FIELDS_X)
        rng_key, splitkey = jax.random.split(rng_key)
        flag_y = jax.random.randint(splitkey, (), 0, self.consts.NUM_FIELDS_Y)

        bombs = jnp.full((n_bombs,), self.consts.PLAYER_STATUS_BOMB, dtype=jnp.int32)
        number_clues = jnp.full((n_number_clues,), self.consts.PLAYER_STATUS_CLUE_PLACEHOLDER_NUMBER, dtype=jnp.int32)
        direction_clues = jnp.full((n_direction_clues,), self.consts.PLAYER_STATUS_CLUE_PLACEHOLDER_DIRECTION,
                                   dtype=jnp.int32)
        filler_fields = jnp.full(
            ((self.consts.NUM_FIELDS_X * self.consts.NUM_FIELDS_Y) - n_bombs - n_number_clues - n_direction_clues,),
            self.consts.PLAYER_STATUS_ALIVE, dtype=jnp.int32)

        field = jnp.concatenate((bombs, number_clues, direction_clues, filler_fields))
        field = jax.random.permutation(splitkey, field)
        field = jnp.reshape(field, (self.consts.NUM_FIELDS_X, self.consts.NUM_FIELDS_Y))
        field = field.at[flag_x, flag_y].set(self.consts.PLAYER_STATUS_FLAG)

        def resolve_number_clue(x, y) -> chex.Array:
            """
            Determine the distance to the flag. This is the larger of the two distances.
            First intuition would be to use the chebyshev distance, but that is not how it's done in the original game

            Args:
                x: x coordinate of the field
                y: y coordinate of the field

            Returns:
                ret: The sprite id of the number clue
            """
            dist_x = jnp.abs(x - flag_x)
            dist_y = jnp.abs(y - flag_y)
            dist_max = jnp.maximum(dist_x, dist_y)
            ret = jnp.reshape(self.consts.PLAYER_STATUS_NUMBER_1 + dist_max - 1, ())
            return ret

        def resolve_direction_clue(x, y) -> chex.Array:
            """
            Determine the direction to the flag. This is done by checking the sign of the difference between the field and the flag.

            Args:
                x: x coordinate of the field
                y: y coordinate of the field
            Returns:
                ret_val: The sprite id of the direction clue
            """
            dx = x - flag_x
            dy = y - flag_y
            ret_val = jax.lax.cond(
                jnp.equal(dx, 0),
                lambda: jax.lax.cond(
                    jnp.greater(dy, 0), lambda: self.consts.PLAYER_STATUS_DIRECTION_UP,
                    lambda: self.consts.PLAYER_STATUS_DIRECTION_DOWN
                ),
                lambda: jax.lax.cond(
                    jnp.equal(dy, 0),
                    lambda: jax.lax.cond(
                        jnp.greater(dx, 0), lambda: self.consts.PLAYER_STATUS_DIRECTION_LEFT,
                        lambda: self.consts.PLAYER_STATUS_DIRECTION_RIGHT
                    ),
                    lambda: jax.lax.cond(
                        jnp.greater(dx * dy, 0),
                        lambda: jax.lax.cond(
                            jnp.greater(dx, 0), lambda: self.consts.PLAYER_STATUS_DIRECTION_UPLEFT,
                            lambda: self.consts.PLAYER_STATUS_DIRECTION_DOWNRIGHT
                        ),
                        lambda: jax.lax.cond(
                            jnp.greater(dx, 0), lambda: self.consts.PLAYER_STATUS_DIRECTION_DOWNLEFT,
                            lambda: self.consts.PLAYER_STATUS_DIRECTION_UPRIGHT
                        )
                    )
                )
            )
            return ret_val

        def resolve_clue(x, y, value):
            """
            Resolves the clue for the given field. This is done by checking if the value is a number clue, direction clue or a different value.
            If the value is a number clue, the distance to the flag is calculated. If the value is a direction clue, the direction to the flag is calculated.
            If the value is a different value, it is returned as is.

            Args:
                x: x coordinate of the field
                y: y coordinate of the field
                value: The value of the field
            Returns:
                value: The resolved value of the field
            """
            return jax.lax.cond(jnp.equal(value, self.consts.PLAYER_STATUS_CLUE_PLACEHOLDER_NUMBER),
                                lambda: resolve_number_clue(x, y),
                                lambda: jax.lax.cond(
                                    jnp.equal(value, self.consts.PLAYER_STATUS_CLUE_PLACEHOLDER_DIRECTION),
                                    lambda: resolve_direction_clue(x, y),
                                    lambda: value
                                    )
                                )

        def vectorized_resolve(x, y):
            """
            Vectorized resolve function to resolve the clues for the given field.

            Args:
                x: x coordinate of the field
                y: y coordinate of the field
            Returns:
                value: The resolved value of the field
            """
            return resolve_clue(x, y, field[x.astype(jnp.int32), y.astype(jnp.int32)])

        field = jnp.fromfunction(vectorized_resolve, (self.consts.NUM_FIELDS_X, self.consts.NUM_FIELDS_Y),
                                 dtype=jnp.int32)

        return field

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: FlagCaptureState) -> FlagCaptureInfo:
        """
        Returns additional information about the game state.
        Args:
            state: The current game state.
        Returns:
            FlagCaptureInfo: Additional information about the game state.
        """
        return FlagCaptureInfo(time=state.time, score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FlagCaptureState) -> bool:
        """
        Returns whether the game is done based on the game state.
        Args:
            state: The current game state.
        """
        return state.time <= 0


class FlagCaptureRenderer(JAXGameRenderer):
    def __init__(self, consts: FlagCaptureConstants = None):
        super().__init__()
        self.consts = consts or FlagCaptureConstants()

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH), # TODO window size?
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND_RASTER,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self._load_sprites()

    def _load_sprites(self):
        """Defines the asset manifest for FlagCapture and loads them using the new utility."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_base = os.path.join(MODULE_DIR, "sprites/flagcapture")

        # Player states: player_0.npy (Normal) to player_18.npy (19 total)
        player_files = [f"player_states/player_{i}.npy" for i in range(19)]

        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            # The SPRITE_PLAYER array (19 states)
            {'name': 'player_states', 'type': 'group', 'files': player_files},
            # The SPRITE_SCORE digits (0-9)
            {'name': 'score_digits', 'type': 'digits', 'pattern': 'green_digits/{}.npy'},
            # The SPRITE_TIMER digits (0-9)
            {'name': 'timer_digits', 'type': 'digits', 'pattern': 'red_digits/{}.npy'},
        ]

        return self.jr.load_and_setup_assets(asset_config, sprite_base)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.
        """
        # 1. Initialize Raster from the pre-rendered Background ID Raster
        raster: jnp.ndarray = self.jr.create_object_raster(self.BACKGROUND_RASTER)

        # 2. Calculate Player Position
        player_x = self.consts.FIELD_PADDING_LEFT + (state.player_x * self.consts.FIELD_WIDTH) + (
                    state.player_x * self.consts.FIELD_GAP_X)
        player_y = self.consts.FIELD_PADDING_TOP + (state.player_y * self.consts.FIELD_HEIGHT) + (
                    state.player_y * self.consts.FIELD_GAP_Y)

        player_offset = self.FLIP_OFFSETS["player_states"]

        # 3. Determine Player Sprite Mask and render (Logic is largely preserved, but calls are updated)

        # Determine the correct sprite index (0-18)
        sprite_idx = jax.lax.cond(jax.lax.eq(state.animation_type, self.consts.ANIMATION_TYPE_NONE),
                                  lambda: jax.lax.cond(jax.lax.eq(state.is_checking, 0),
                                                       lambda: 0,  # Index 0: Normal (SPRITE_PLAYER[0])
                                                       lambda: state.field[state.player_x, state.player_y]
                                                       # Index N: Clue/Bomb/Flag (SPRITE_PLAYER[N])
                                                       ),
                                  lambda: jax.lax.cond(
                                      jax.lax.eq(state.animation_type, self.consts.ANIMATION_TYPE_EXPLOSION),
                                      lambda: jax.lax.cond(
                                          jax.lax.lt(jnp.mod(state.animation_cooldown, 8), 4),
                                          lambda: 1,  # Index 1: Explosion (SPRITE_PLAYER[1]) - Blinking
                                          lambda: -1),  # Invisible/Transparent
                                      lambda: jax.lax.cond(
                                          jax.lax.eq(state.animation_type, self.consts.ANIMATION_TYPE_FLAG),
                                          lambda: jax.lax.cond(
                                              jax.lax.lt(jnp.mod(state.animation_cooldown, 20), 10),
                                              lambda: 2,  # Index 2: Flag (SPRITE_PLAYER[2]) - Blinking
                                              lambda: -1),  # Invisible/Transparent
                                          lambda: -1)  # Default invisible
                                      )
                                  )

        # Draw the Player sprite only if the index is valid (not -1 for transparent)
        raster = jax.lax.cond(
            sprite_idx >= 0,
            lambda r: self.jr.render_at(
                r,
                player_x.astype(jnp.int32),
                player_y.astype(jnp.int32),
                self.SHAPE_MASKS["player_states"][sprite_idx],  # Use the ID mask
                flip_offset=player_offset
            ),
            lambda r: r,  # Do nothing if transparent
            operand=raster
        )

        # 4. Render Header (Score and Timer)
        # The spacing is 16, max_digits=3 based on original code structure
        raster = self._render_header(
            state.score,
            raster,
            self.SHAPE_MASKS["score_digits"],
            single_digit_x=32,
            double_digit_x=16,
            y=self.consts.SCORE_AND_TIMER_PADDING_TOP,
            max_digits=3
        )

        raster = self._render_header(
            state.time // self.consts.STEPS_PER_SECOND,
            raster,
            self.SHAPE_MASKS["timer_digits"],
            single_digit_x=112,
            double_digit_x=96,
            y=self.consts.SCORE_AND_TIMER_PADDING_TOP,
            max_digits=3
        )

        # 5. Final Palette Lookup
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7))
    def _render_header(self, number, raster, digit_masks, single_digit_x, double_digit_x, y, max_digits):
        """Uses the new utility methods to render a score/timer header."""

        digits = self.jr.int_to_digits(number, max_digits=max_digits)

        num_to_render = jnp.where(number < 10, 1, jnp.where(number < 100, 2, max_digits))

        start_index = max_digits - num_to_render

        render_x = jnp.where(num_to_render == 1, single_digit_x, double_digit_x)

        # Use the existing selective label renderer with korrekten Indizes
        raster = self.jr.render_label_selective(
            raster,
            render_x,
            y,
            digits,
            digit_masks,  # This is the full stack of ID masks (0-9)
            start_index,
            num_to_render,
            spacing=16,
            max_digits_to_render=max_digits
        )
        return raster

