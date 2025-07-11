import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

#TODO: Constants

# Tetris pieces
TETROMINOS = jnp.array([
    [  # I
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]
    ],
    [  # O
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]
    ] * 4,
    [  # T
        [[0,0,0,0],[0,1,0,0],[1,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]]
    ],
    [  # S
        [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]]
    ] * 2,
    [  # Z
        [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]]
    ] * 2,
    [  # J
        [[0,0,0,0],[1,0,0,0],[1,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]]
    ],
    [  # L
        [[0,0,0,0],[0,0,1,0],[1,1,1,0],[0,0,0,0]],
        [[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]]
    ]
], dtype=jnp.int32)

# Constants Board
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# immutable state container
class TetrisState(NamedTuple):
    board: chex.Array           # Game Board, height x width
    #TODO: how much pieces left?
    current_piece_x: chex.Array
    current_piece_y: chex.Array
    current_piece_shape: chex.Array         # given in index
    current_piece_rotation: chex.Array      # given in index
    next_piece_shape: chex.Array            # given in index
    score: chex.Array

class PieceInfo(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    shape: jnp.ndarray
    rotation: jnp.ndarray

class TetrisObservation(NamedTuple):
    board: chex.Array
    current_piece: PieceInfo
    next_piece: chex.Array
    score: chex.Array

class TetrisInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array #TODO: for what?

@jax.jit
# TODO: logic about human Input
# TODO: use x + y or just wrapper?
# TODO: implement is_valid
def player_step(state: TetrisState, action: chex.Array) -> TetrisState:
    # 1. handle rotation
    new_rotation = jax.lax.cond(
        action == Action.FIRE,
        lambda r: (r+1) % 4,
        lambda r: r,
        operand = state.current_piece_rotation,
        # if action == Action.FIRE:
        #     new_rotation = (rotation + 1) % 4
        # else:
        #     new_rotation = rotation
    )

    # 2. handle left+right
    left = action == Action.LEFT
    right = action == Action.RIGHT
    delta_x = jnp.where(left, -1, jnp.where(right, 1, 0))
    # if move_left:
    #    delta_x = -1
    # elif move_right:
    #    delta_x = 1
    # else:
    #    delta_x = 0
    new_x = state.current_piece_x + delta_x

    # 3. check move+rotation validity
    is_move_valid = is_valid(state.board, new_x, state.current_piece_y, state.current_piece_shape, new_rotation)
    # if illegal then restore
    x = jax.lax.select(is_move_valid, new_x, state.current_piece_x)
    rotation = jax.lax.select(is_move_valid, new_rotation, state.current_piece_rotation)

    # 4. move down
    delta_y = jnp.where(action == Action.DOWN, 2, 1)
    new_y = state.current_piece_y + delta_y
    is_down_valid = is_valid(state.board, x, new_y , state.current_piece_shape, rotation) #TODO: new or old?



def piece_step(
    # TODO: logic about falling down and collision check

):

# no Enemy_step

@jax.jit
def _reset(

):
    #TODO

class JaxTetris(JaxEnvironment[TetrisState, TetrisObservation, TetrisInfo]):
    def __init__(self, reward_funcs: list[callable]=None):
        super().__init__()
        self.renderer = TetrisRenderer()#TODO
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN
        ]
        self.obs_size = #TODO

    def reset(self, key=None) -> Tuple[TetrisObservation, TetrisState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward.
        """
        state = TetrisState(
            #TODO
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TetrisState, action: chex.Array) -> Tuple[TetrisObservation, TetrisState, float, bool, TetrisInfo]: #TODO: why float+bool?

    def render(self, state: TetrisState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TetrisState):
        #TODO


    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TetrisObservation) -> jnp.ndarray:
        return jnp.concatenate([
            #TODO
        ])

    def action_space(self) -> spaces.Discrete:
        """
        Returns the action space for Tetris.
        Actions are:
        0: NOOP
        1: FIRE
        2: RIGHT
        3: LEFT
        4: DOWN
        """
        return spaces.Discrete(5)

    def observation_space(self) -> spaces:
        """
        Returns the observation space for Tetris.
        The Observation contains:
         # TODO:
        """
        return spaces.Dict({
            #TODO
        })

    def image_space(self) -> spaces.Box:
        """
        Returns the image space for Tetris.
        """
        return spaces.Box(
            #TODO
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TetrisState, all_rewards: chex.Array) -> TetrisInfo:
        #TODO


    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: TetrisState, state: TetrisState):
        #TODO



    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: TetrisState, state: TetrisState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TetrisState) -> bool:
        #TODO

def load_sprites():
    """
    Load all sprites required for Tetris rendering.
    """
    #TODO

class TetrisRenderer(JAXGameRenderer):
    """
    JAX-based Tetris game renderer, optimized with JIT compilation.
    """

    def __init__(self):
        #TODO

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
            Renders the current game state using JAX operations.

            Args:
                state: A PongState object containing the current game state.

            Returns:
                A JAX array representing the rendered frame.
        """
        #TODO
