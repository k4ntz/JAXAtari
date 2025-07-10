import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame

from jaxatari.environment import JAXAtariAction

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

#
# by Tim Morgner and Jan Larionow
#

# Note
# For us, play.py does not work consistently and sometimes freezes. However, based on the Mattermost
# messages, we assume that it works for you. We have not modified play.py.

# region Constants
# Constants for game environment
WIDTH = 160
HEIGHT = 210
# endregion

# region Pygame Constants
# Pygame window dimensions
SCALING_FACTOR = 3
WINDOW_WIDTH = WIDTH * SCALING_FACTOR
WINDOW_HEIGHT = HEIGHT * SCALING_FACTOR
# endregion

# region Player Status
# Define player status (these are also used as field types)
PLAYER_STATUS_ALIVE = 0
PLAYER_STATUS_BOMB = 1
PLAYER_STATUS_FLAG = 2
PLAYER_STATUS_NUMBER_1 = 3
PLAYER_STATUS_NUMBER_2 = 4
PLAYER_STATUS_NUMBER_3 = 5
PLAYER_STATUS_NUMBER_4 = 6
PLAYER_STATUS_NUMBER_5 = 7
PLAYER_STATUS_NUMBER_6 = 8
PLAYER_STATUS_NUMBER_7 = 9
PLAYER_STATUS_NUMBER_8 = 10
PLAYER_STATUS_DIRECTION_UP = 11
PLAYER_STATUS_DIRECTION_RIGHT = 12
PLAYER_STATUS_DIRECTION_DOWN = 13
PLAYER_STATUS_DIRECTION_LEFT = 14
PLAYER_STATUS_DIRECTION_UPRIGHT = 15
PLAYER_STATUS_DIRECTION_UPLEFT = 16
PLAYER_STATUS_DIRECTION_DOWNRIGHT = 17
PLAYER_STATUS_DIRECTION_DOWNLEFT = 18
PLAYER_STATUS_CLUE_PLACEHOLDER_NUMBER = 19
PLAYER_STATUS_CLUE_PLACEHOLDER_DIRECTION = 20

# endregion

# region Game Field Constants
# Define game field constants
NUM_FIELDS_X = 9
NUM_FIELDS_Y = 7
# endregion

# region Field Padding and Gaps
# Define field padding and gaps
FIELD_PADDING_LEFT = 12
FIELD_PADDING_TOP = 11
FIELD_GAP_X = 8
FIELD_GAP_Y = 4
FIELD_WIDTH = FIELD_HEIGHT = 8
NUMBER_WIDTH = 12
NUMBER_HEIGHT = 5
# endregion

# region Game Constants
# Define game constants
NUM_BOMBS = 3
NUM_NUMBER_CLUES = 30
NUM_DIRECTION_CLUES = 30
MOVE_COOLDOWN = 15
STEPS_PER_SECOND = 30
# endregion

# region Animation Constants
# Define animation constants
ANIMATION_TYPE_NONE = 0
ANIMATION_TYPE_EXPLOSION = 1
ANIMATION_TYPE_FLAG = 2


class FlagCaptureState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    time: chex.Array
    is_checking: chex.Array
    score: chex.Array
    field: chex.Array  # Shape (9, 7)
    player_move_cooldown: chex.Array
    animation_cooldown: chex.Array
    animation_type: chex.Array
    rng_key: chex.PRNGKey


class PlayerEntity(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array
    status: chex.Array


class FlagCaptureObservation(NamedTuple):
    player: PlayerEntity
    score: chex.Array


class FlagCaptureInfo(NamedTuple):
    time: chex.Array
    score: chex.Array
    all_rewards: chex.Array


def generate_field(rng_key, n_bombs, n_number_clues, n_direction_clues):
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
    flag_x = jax.random.randint(splitkey, (), 0, NUM_FIELDS_X)
    rng_key, splitkey = jax.random.split(rng_key)
    flag_y = jax.random.randint(splitkey, (), 0, NUM_FIELDS_Y)

    bombs = jnp.full((n_bombs,), PLAYER_STATUS_BOMB, dtype=jnp.int32)
    number_clues = jnp.full((n_number_clues,), PLAYER_STATUS_CLUE_PLACEHOLDER_NUMBER, dtype=jnp.int32)
    direction_clues = jnp.full((n_direction_clues,), PLAYER_STATUS_CLUE_PLACEHOLDER_DIRECTION, dtype=jnp.int32)
    # This is only to handle possibly misconfigured constants
    filler_fields = jnp.full(((NUM_FIELDS_X * NUM_FIELDS_Y) - n_bombs - n_number_clues - n_direction_clues,),
                             PLAYER_STATUS_ALIVE, dtype=jnp.int32)

    field = jnp.concatenate((bombs, number_clues, direction_clues, filler_fields))
    field = jax.random.permutation(splitkey, field)
    field = jnp.reshape(field, (NUM_FIELDS_X, NUM_FIELDS_Y))
    field = field.at[flag_x, flag_y].set(PLAYER_STATUS_FLAG)  # This may override the bomb, but that is not a problem

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
        ret = jnp.reshape(PLAYER_STATUS_NUMBER_1 + dist_max - 1, ())
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
                jnp.greater(dy, 0), lambda: PLAYER_STATUS_DIRECTION_UP, lambda: PLAYER_STATUS_DIRECTION_DOWN
            ),
            lambda: jax.lax.cond(
                jnp.equal(dy, 0),
                lambda: jax.lax.cond(
                    jnp.greater(dx, 0), lambda: PLAYER_STATUS_DIRECTION_LEFT, lambda: PLAYER_STATUS_DIRECTION_RIGHT
                ),
                lambda: jax.lax.cond(
                    jnp.greater(dx * dy, 0),
                    lambda: jax.lax.cond(
                        jnp.greater(dx, 0), lambda: PLAYER_STATUS_DIRECTION_UPLEFT,
                        lambda: PLAYER_STATUS_DIRECTION_DOWNRIGHT
                    ),
                    lambda: jax.lax.cond(
                        jnp.greater(dx, 0), lambda: PLAYER_STATUS_DIRECTION_DOWNLEFT,
                        lambda: PLAYER_STATUS_DIRECTION_UPRIGHT
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
        return jax.lax.cond(jnp.equal(value, PLAYER_STATUS_CLUE_PLACEHOLDER_NUMBER),
                            lambda: resolve_number_clue(x, y),
                            lambda: jax.lax.cond(jnp.equal(value, PLAYER_STATUS_CLUE_PLACEHOLDER_DIRECTION),
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

    # This calls vectorized_resolve for every cell in the field and returns a new field from the result
    field = jnp.fromfunction(vectorized_resolve, (NUM_FIELDS_X, NUM_FIELDS_Y), dtype=jnp.int32)
    # jax.debug.print("field after:\n{field}", field=field)

    return field


@jax.jit
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
    # check if the player is firing(checking). This is any action like FIRE, UPFIRE, DOWNFIRE, etc.
    # do this by checking if the action is 1 (FIRE) or between 10 and 17 (UPFIRE, DOWNFIRE, etc.)
    new_is_checking = jax.lax.cond(jnp.logical_or(jnp.equal(action, JAXAtariAction.FIRE),
                                                  jnp.logical_and(jnp.greater_equal(action, JAXAtariAction.UPFIRE),
                                                                  jnp.less_equal(action, JAXAtariAction.DOWNLEFTFIRE))),
                                   lambda: 1, lambda: 0)
    # check if the player is moving upwards. This is any action like UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
    is_up = jnp.logical_or(jnp.equal(action, JAXAtariAction.UP),
                           jnp.logical_or(jnp.equal(action, JAXAtariAction.UPFIRE),
                                          jnp.logical_or(jnp.equal(action, JAXAtariAction.UPRIGHT),
                                                         jnp.logical_or(jnp.equal(action, JAXAtariAction.UPLEFT),
                                                                        jnp.logical_or(jnp.equal(action,
                                                                                                 JAXAtariAction.UPRIGHTFIRE),
                                                                                       jnp.equal(action,
                                                                                                 JAXAtariAction.UPLEFTFIRE))))))
    is_down = jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWN),
                             jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNFIRE),
                                            jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNRIGHT),
                                                           jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNLEFT),
                                                                          jnp.logical_or(jnp.equal(action,
                                                                                                   JAXAtariAction.DOWNRIGHTFIRE),
                                                                                         jnp.equal(action,
                                                                                                   JAXAtariAction.DOWNLEFTFIRE))))))
    is_left = jnp.logical_or(jnp.equal(action, JAXAtariAction.LEFT),
                             jnp.logical_or(jnp.equal(action, JAXAtariAction.UPLEFT),
                                            jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNLEFT),
                                                           jnp.logical_or(jnp.equal(action, JAXAtariAction.LEFTFIRE),
                                                                          jnp.logical_or(jnp.equal(action,
                                                                                                   JAXAtariAction.UPLEFTFIRE),
                                                                                         jnp.equal(action,
                                                                                                   JAXAtariAction.DOWNLEFTFIRE))))))
    is_right = jnp.logical_or(jnp.equal(action, JAXAtariAction.RIGHT),
                              jnp.logical_or(jnp.equal(action, JAXAtariAction.UPRIGHT),
                                             jnp.logical_or(jnp.equal(action, JAXAtariAction.DOWNRIGHT),
                                                            jnp.logical_or(jnp.equal(action, JAXAtariAction.RIGHTFIRE),
                                                                           jnp.logical_or(jnp.equal(action,
                                                                                                    JAXAtariAction.UPRIGHTFIRE),
                                                                                          jnp.equal(action,
                                                                                                    JAXAtariAction.DOWNRIGHTFIRE))))))

    # if player is moving down add 1 to player_y
    new_player_y = jax.lax.cond(is_down, lambda: player_y + 1, lambda: player_y)
    # if player is moving up subtract 1 from player_y
    new_player_y = jax.lax.cond(is_up, lambda: player_y - 1, lambda: new_player_y)
    # if player is moving left subtract 1 from player_x
    new_player_x = jax.lax.cond(is_left, lambda: player_x - 1, lambda: player_x)
    # if player is moving right add 1 to player_x
    new_player_x = jax.lax.cond(is_right, lambda: player_x + 1, lambda: new_player_x)
    # modulo the player_x and player_y to be on the field
    # This adds the movement cooldown, border wrapping and prevents moving while checking or animating
    new_player_x = jax.lax.cond(jnp.logical_or(new_is_checking,
                                               jnp.logical_or(jnp.not_equal(animation_type, ANIMATION_TYPE_NONE),
                                                              jnp.greater(player_move_cooldown, 0))), lambda: player_x,
                                lambda: jnp.mod(new_player_x, NUM_FIELDS_X))
    new_player_y = jax.lax.cond(jnp.logical_or(new_is_checking,
                                               jnp.logical_or(jnp.not_equal(animation_type, ANIMATION_TYPE_NONE),
                                                              jnp.greater(player_move_cooldown, 0))), lambda: player_y,
                                lambda: jnp.mod(new_player_y, NUM_FIELDS_Y))

    # if cooldown is <= 0 set it to MOVE_COOLDOWN else subtract 1
    new_player_move_cooldown = jax.lax.cond(jnp.less_equal(player_move_cooldown, 0), lambda: MOVE_COOLDOWN,
                                            lambda: player_move_cooldown - 1)

    return new_player_x, new_player_y, new_is_checking, new_player_move_cooldown


class JaxFlagCapture(JaxEnvironment[FlagCaptureState, FlagCaptureObservation, FlagCaptureInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            JAXAtariAction.NOOP,
            JAXAtariAction.FIRE,
            JAXAtariAction.UP,
            JAXAtariAction.RIGHT,
            JAXAtariAction.LEFT,
            JAXAtariAction.DOWN,
            JAXAtariAction.UPRIGHT,
            JAXAtariAction.UPLEFT,
            JAXAtariAction.DOWNRIGHT,
            JAXAtariAction.DOWNLEFT,
            JAXAtariAction.UPFIRE,
            JAXAtariAction.RIGHTFIRE,
            JAXAtariAction.LEFTFIRE,
            JAXAtariAction.DOWNFIRE,
            JAXAtariAction.UPRIGHTFIRE,
            JAXAtariAction.UPLEFTFIRE,
            JAXAtariAction.DOWNRIGHTFIRE,
            JAXAtariAction.DOWNLEFTFIRE
        }

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
        generated_field = generate_field(splitkey, NUM_BOMBS, NUM_NUMBER_CLUES, NUM_DIRECTION_CLUES)
        state = FlagCaptureState(
            player_x=jnp.array(0).astype(jnp.int32),
            player_y=jnp.array(0).astype(jnp.int32),
            time=jnp.array(75 * STEPS_PER_SECOND).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            is_checking=jnp.array(1).astype(jnp.int32),
            player_move_cooldown=jnp.array(0).astype(jnp.int32),
            animation_cooldown=jnp.array(0).astype(jnp.int32),
            animation_type=jnp.array(0).astype(jnp.int32),
            field=generated_field,
            rng_key=newkey,
        )
        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FlagCaptureState):
        """
        Returns the observation of the game state.
        Args:
            state: The current game state.
        Returns:
            FlagCaptureObservation: The observation of the game state.
        """
        # create player
        return FlagCaptureObservation(
            player=PlayerEntity(
                x=jnp.array(
                    FIELD_PADDING_LEFT + (state.player_x * FIELD_WIDTH) + (state.player_x * FIELD_GAP_X)).astype(
                    jnp.int32),
                y=jnp.array(
                    FIELD_PADDING_TOP + (state.player_y * FIELD_HEIGHT) + (state.player_y * FIELD_GAP_Y)).astype(
                    jnp.int32),
                width=jnp.array(FIELD_WIDTH).astype(jnp.int32),
                height=jnp.array(FIELD_HEIGHT).astype(jnp.int32),
                status=jnp.array(PLAYER_STATUS_ALIVE).astype(jnp.int32),
            ),
            score=state.score
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FlagCaptureState, action: chex.Array) -> Tuple[
        FlagCaptureObservation, FlagCaptureState, float, bool, FlagCaptureInfo]:
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

        new_player_x, new_player_y, new_is_checking, new_player_move_cooldown = player_step(
            state.player_x,
            state.player_y,
            state.is_checking,
            state.player_move_cooldown,
            state.animation_type,
            action,
        )

        # Check if the animation cooldown is less or equal to 0 and a animation type is set
        # If the animation type is bomb, reset the player position
        # If the animation type is flag, reset the player position, set the score to 1 and generate a new field
        bomb_animation_over = jnp.logical_and(
            jnp.less_equal(state.animation_cooldown, 0),
            jnp.equal(state.animation_type, ANIMATION_TYPE_EXPLOSION)
        )
        flag_animation_over = jnp.logical_and(
            jnp.less_equal(state.animation_cooldown, 0),
            jnp.equal(state.animation_type, ANIMATION_TYPE_FLAG)
        )

        # Increment the score if the animation is over and the animation type is flag
        new_score = jax.lax.cond(
            flag_animation_over,
            lambda: state.score + 1,
            lambda: state.score
        )
        rng_key, splitkey = jax.random.split(state.rng_key)
        new_field = jax.lax.cond(
            flag_animation_over,
            lambda: generate_field(splitkey, NUM_BOMBS, NUM_NUMBER_CLUES, NUM_DIRECTION_CLUES),
            lambda: state.field
        )

        new_player_x = jax.lax.cond(jnp.logical_or(bomb_animation_over, flag_animation_over),
                                    lambda: 0,
                                    lambda: new_player_x)
        new_player_y = jax.lax.cond(jnp.logical_or(bomb_animation_over, flag_animation_over),
                                    lambda: 0,
                                    lambda: new_player_y)

        # Check if the player is checking (firing) and if the current field is a bomb or a flag (only if the animation_type is currently none)
        # If the player is checking and the field is a bomb or flag, set the animation type

        new_animation_type = jax.lax.cond(
            jnp.logical_or(bomb_animation_over, flag_animation_over),
            lambda: ANIMATION_TYPE_NONE,
            lambda: jax.lax.cond(
                jnp.logical_and(jnp.equal(state.field[new_player_x, new_player_y], PLAYER_STATUS_BOMB),
                                jnp.logical_and(
                                    jnp.equal(new_is_checking, 1),
                                    jnp.equal(state.animation_type, ANIMATION_TYPE_NONE))),
                lambda: ANIMATION_TYPE_EXPLOSION,
                lambda: jax.lax.cond(
                    jnp.logical_and(jnp.equal(state.field[new_player_x, new_player_y], PLAYER_STATUS_FLAG),
                                    jnp.logical_and(
                                        jnp.equal(new_is_checking, 1),
                                        jnp.equal(state.animation_type, ANIMATION_TYPE_NONE))),
                    lambda: ANIMATION_TYPE_FLAG,
                    lambda: state.animation_type))
        )

        new_animation_cooldown = jax.lax.cond(
            jnp.logical_and(jnp.equal(state.animation_type, ANIMATION_TYPE_NONE),
                            jnp.equal(new_animation_type, ANIMATION_TYPE_EXPLOSION)),
            lambda: 15,
            lambda: jax.lax.cond(
                jnp.logical_and(jnp.equal(state.animation_type, ANIMATION_TYPE_NONE),
                                jnp.equal(new_animation_type, ANIMATION_TYPE_FLAG)),
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
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        """
        Returns the action space of the game environment.
        Returns:
            action_space: The action space of the game environment.
        """
        return jnp.array(list(self.action_set), dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: FlagCaptureState, all_rewards: chex.Array) -> FlagCaptureInfo:
        """
        Returns additional information about the game state.
        Args:
            state: The current game state.
            all_rewards: The rewards received after taking the action.
        Returns:
            FlagCaptureInfo: Additional information about the game state.
        """
        return FlagCaptureInfo(time=state.time, all_rewards=all_rewards, score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: FlagCaptureState, state: FlagCaptureState):
        """
        Returns the environment reward based on the game state.
        Args:
            previous_state: The previous game state.
        """
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: FlagCaptureState, state: FlagCaptureState):
        """
        Returns all rewards based on the game state.
        Args:
            previous_state: The previous game state.
            state: The current game state.
        Returns:
            rewards: The rewards received after taking the action.
        """
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FlagCaptureState) -> bool:
        """
        Returns whether the game is done based on the game state.
        Args:
            state: The current game state.
        """
        return state.time <= 0


def load_sprites():
    """
    Load all sprites required for Flag Capture rendering.
    Returns:
        SPRITE_BG: Background sprite.
        SPRITE_PLAYER: Player sprites.
        SPRITE_SCORE: Score sprites.
        SPRITE_TIMER: Timer sprites.
    """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/flagcapture/background.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(background, axis=0)
    SPRITE_PLAYER = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/flagcapture/player_states/player_{}.npy"),
                                           num_chars=19)
    SPRITE_SCORE = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/flagcapture/green_digits/{}.npy"),
                                          num_chars=10)
    SPRITE_TIMER = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/flagcapture/red_digits/{}.npy"),
                                          num_chars=10)

    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_SCORE,
        SPRITE_TIMER,
    )


class FlagCaptureRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_SCORE,
            self.SPRITE_TIMER,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A FlagCaptureState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster: jnp.ndarray = jnp.zeros((WIDTH, HEIGHT, 3))

        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Draw the player
        player_x = FIELD_PADDING_LEFT + (state.player_x * FIELD_WIDTH) + (state.player_x * FIELD_GAP_X)
        player_y = FIELD_PADDING_TOP + (state.player_y * FIELD_HEIGHT) + (state.player_y * FIELD_GAP_Y)

        # Wenn keine Animation läuft:
        # Wenn is_checking == 1, dann ist die Spieler Sprite >= 1 (Bombe, Flag, Zahl, ect.)
        # Die genaue Sprite wird durch den wert von field[x][y] bestimmt
        # Ansonsten ist die Spieler Sprite == 0 aka Männchen
        # Wenn eine Animation läuft dann bei jedem ungraden animation_cooldown frame nichts rendern (das ist die "animation")
        raster = jax.lax.cond(jax.lax.eq(state.animation_type, ANIMATION_TYPE_NONE),
                              lambda: jax.lax.cond(jax.lax.eq(state.is_checking, 0),
                                                   lambda: aj.render_at(raster, player_x, player_y,
                                                                        self.SPRITE_PLAYER[0]),
                                                   lambda: aj.render_at(raster, player_x, player_y,
                                                                        self.SPRITE_PLAYER[
                                                                            state.field[
                                                                                state.player_x, state.player_y]]),
                                                   ),
                              lambda: jax.lax.cond(jax.lax.eq(state.animation_type, ANIMATION_TYPE_EXPLOSION),
                                                   lambda: jax.lax.cond(
                                                       jax.lax.lt(jnp.mod(state.animation_cooldown, 8), 4),
                                                       lambda: aj.render_at(raster, player_x, player_y,
                                                                            self.SPRITE_PLAYER[1]),
                                                       lambda: raster),
                                                   lambda: jax.lax.cond(
                                                       jax.lax.eq(state.animation_type, ANIMATION_TYPE_FLAG),
                                                       lambda: jax.lax.cond(
                                                           jax.lax.lt(jnp.mod(state.animation_cooldown, 20), 10),
                                                           lambda: aj.render_at(raster, player_x, player_y,
                                                                                self.SPRITE_PLAYER[2]),
                                                           lambda: raster),
                                                       lambda: raster),
                                                   )
                              )

        raster = render_header(state.score, raster, self.SPRITE_SCORE,
                               32, 16,
                               3)

        raster = render_header(state.time // STEPS_PER_SECOND, raster, self.SPRITE_TIMER, 112, 96, 3)

        return raster


def render_header(number, raster, sprites, single_digit_x, double_digit_x, y):
    """
    Renders the header (score or timer) on the screen.
    Args:
        number: The number to be rendered (score or timer).
        raster: The raster to render on.
        sprites: The sprite array for rendering the digits.
        single_digit_x: X position for single digit rendering.
        double_digit_x: X position for double digit rendering.
        y: Y position for rendering.
    """
    # 1. Get digit arrays (always 2 digits)
    digits = aj.int_to_digits(number, max_digits=2)

    # 2. Determine parameters for timer rendering using jax.lax.select
    is_single_digit = number < 10
    start_index = jax.lax.select(is_single_digit, 1, 0)  # Start at index 1 if single, 0 if double
    num_to_render = jax.lax.select(is_single_digit, 1, 2)  # Render 1 digit if single, 2 if double
    # Adjust X position: If single digit, center it slightly by moving right by one spacing
    render_x = jax.lax.select(is_single_digit,
                              single_digit_x,
                              double_digit_x)

    # 3. Render player score using the selective renderer
    raster = aj.render_label_selective(raster, render_x, y,
                                       digits, sprites,
                                       start_index, num_to_render,
                                       spacing=16)

    return raster


def get_human_action() -> chex.Array:
    """
    Records human input for the game.

    Returns:
        action: int, action taken by the player (FIRE, LEFT, RIGHT, UPRIGHT, FIREDOWNRIGHT, etc.)
    """
    keys = pygame.key.get_pressed()
    # pygame.K_a  Links
    # pygame.K_d  Rechts
    # pygame.K_w  Hoch
    # pygame.K_s  Runter
    # pygame.K_SPACE  Aufdecken (fire)

    pressed_buttons = 0
    if keys[pygame.K_a]:
        pressed_buttons += 1
    if keys[pygame.K_d]:
        pressed_buttons += 1
    if keys[pygame.K_w]:
        pressed_buttons += 1
    if keys[pygame.K_s]:
        pressed_buttons += 1
    if pressed_buttons > 3:
        print("You have pressed a physically impossible combination of buttons")
        return jnp.array(JAXAtariAction.NOOP)

    if keys[pygame.K_SPACE]:
        # All actions with fire
        if keys[pygame.K_w] and keys[pygame.K_a]:
            return jnp.array(JAXAtariAction.UPLEFTFIRE)
        elif keys[pygame.K_w] and keys[pygame.K_d]:
            return jnp.array(JAXAtariAction.UPRIGHTFIRE)
        elif keys[pygame.K_s] and keys[pygame.K_a]:
            return jnp.array(JAXAtariAction.DOWNLEFTFIRE)
        elif keys[pygame.K_s] and keys[pygame.K_d]:
            return jnp.array(JAXAtariAction.DOWNRIGHTFIRE)
        elif keys[pygame.K_w]:
            return jnp.array(JAXAtariAction.UPFIRE)
        elif keys[pygame.K_a]:
            return jnp.array(JAXAtariAction.LEFTFIRE)
        elif keys[pygame.K_d]:
            return jnp.array(JAXAtariAction.RIGHTFIRE)
        elif keys[pygame.K_s]:
            return jnp.array(JAXAtariAction.DOWNFIRE)
        else:
            return jnp.array(JAXAtariAction.FIRE)
    else:
        # All actions without fire
        if keys[pygame.K_w] and keys[pygame.K_a]:
            return jnp.array(JAXAtariAction.UPLEFT)
        elif keys[pygame.K_w] and keys[pygame.K_d]:
            return jnp.array(JAXAtariAction.UPRIGHT)
        elif keys[pygame.K_s] and keys[pygame.K_a]:
            return jnp.array(JAXAtariAction.DOWNLEFT)
        elif keys[pygame.K_s] and keys[pygame.K_d]:
            return jnp.array(JAXAtariAction.DOWNRIGHT)
        elif keys[pygame.K_w]:
            return jnp.array(JAXAtariAction.UP)
        elif keys[pygame.K_a]:
            return jnp.array(JAXAtariAction.LEFT)
        elif keys[pygame.K_d]:
            return jnp.array(JAXAtariAction.RIGHT)
        elif keys[pygame.K_s]:
            return jnp.array(JAXAtariAction.DOWN)
        else:
            return jnp.array(JAXAtariAction.NOOP)


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flag Capture Game")
    clock = pygame.time.Clock()

    game = JaxFlagCapture()

    # Create the JAX renderer
    renderer = FlagCaptureRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                    event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(30)

    pygame.quit()
