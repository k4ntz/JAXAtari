#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX Casino Poker Solitaire
#
# Simulates the Atari Casino Poker Solitaire game
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember

from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from jax_casino_renderer import CasinoRenderer  # pylint: disable=import-error

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.spaces import Box, Dict, Discrete, Space


class CasinoPokerSolitaireConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    INITIAL_PLAYER_SCORE = jnp.array(0).astype(jnp.int32)  # starts with 0
    CARD_VALUES = jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    NUM_CARDS_IN_DECK = 52
    """
    Scoring:
      Royal Flush       500     Three of a Kind     60
      Straight Flush    300     Flush               50
      Four of a Kind    160     Two Pair            30
      Straight          120     One Pair            10
      Full House        100     High Card           0
    12 poker hands evaluated, 5 in a row, 5 in a column, 2 diagonals
    """


class CasinoPokerSolitaireState(NamedTuple):
    # colors:   D -> diamonds ♦
    #           C -> clubs ♣
    #           H -> hearts ♥
    #           S -> spades ♠
    # -1 -> empty card
    # 0 -> no card currently
    # Values:   01 -> D2            14 -> C2            27 -> H2            40 -> S2
    #           02 -> D3            15 -> C3            28 -> H3            41 -> S3
    #           03 -> D4            16 -> C4            29 -> H4            42 -> S4
    #           ...                 ...                 ...                 ...
    #           09 -> D10           22 -> C10           35 -> H10           48 -> S10
    #           10 -> DJ            23 -> CJ            36 -> HJ            49 -> SJ
    #           11 -> DK            24 -> CK            37 -> HK            50 -> SK
    #           12 -> DQ            25 -> CQ            38 -> HQ            51 -> SQ
    #           13 -> DA            26 -> CA            39 -> HA            52 -> SA

    key: jax.random.PRNGKey
    step_counter: jnp.ndarray
    state_counter: jnp.ndarray  # Manages the game's state machine

    player_score: jnp.ndarray
    """ The score of the player """
    player_round_score: jnp.ndarray
    """ The final score of the current round """
    staple: jnp.ndarray
    """ The staple of cards to draw from """
    current_card: jnp.ndarray
    """ The current card to place """
    dealt_cards: jnp.ndarray
    """ The number of dealt cards """
    board: jnp.ndarray
    """ The 5x5 board where cards are placed """
    # -1 -> empty cell
    # index 0-4 first row, index 5-9 second row, ...
    cursor_pos_x: jnp.ndarray
    cursor_pos_y: jnp.ndarray


class CasinoPokerSolitaireObservation(NamedTuple):
    player_score: jnp.ndarray
    player_round_score: jnp.ndarray
    current_card: jnp.ndarray
    board: jnp.ndarray
    cursor_pos_x: jnp.ndarray
    cursor_pos_y: jnp.ndarray


class CasinoPokerSolitaireInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: jnp.ndarray


@jax.jit
def calculate_hand_value(hand: chex.Array, consts):
    """Calculates the card value of all cards in hand"""
    # check for two/three/four of a kind, full house
    # check for flush
    # check for straight (-> straight flush -> royal flush)

    hand_values = consts.CARD_VALUES[(hand - 1) % 13]  # values without color
    counts = jnp.array(
        [jnp.sum(hand_values == v) for v in range(0, 13)]
    )  # count occurrences of each value
    counts = jnp.sort(counts)[::-1]  # sort descending counts
    is_flush = jnp.all((hand - 1) // 13 == (hand[0] - 1) // 13)  # TMP: ok
    sorted_values = jnp.sort(hand_values)  # TMP: ok
    is_straight = jnp.all(sorted_values[1:] - sorted_values[:-1] == 1) & (
        sorted_values[-1] - sorted_values[0] == len(sorted_values) - 1
    )
    is_royal = jnp.array_equal(
        jnp.sort(hand_values), jnp.array([10, 11, 12, 13, 14])
    )  # 10, J, Q, K, A
    score = jnp.select(
        [
            (is_straight & is_flush & is_royal),
            (is_straight & is_flush),
            (counts[0] == 4),
            (counts[0] == 3) & (counts[1] == 2),
            is_flush,
            is_straight,
            (counts[0] == 3),
            (counts[0] == 2) & (counts[1] == 2),
            (counts[0] == 2),
        ],
        [500, 300, 160, 100, 50, 120, 60, 30, 10],
        default=0,
    )

    return score


@jax.jit
def calculate_total_score(board: chex.Array, consts):
    """Calculates the total score of the board"""
    # get rows, cols and diagonals; execute calculate_hand_value
    rows = board.reshape((5, 5))
    cols = board.reshape((5, 5)).T
    diagonal1 = jnp.array([board[i * 6] for i in range(5)])  # 0, 6, 12, 18, 24
    diagonal2 = jnp.array([board[i * 4 + 4] for i in range(5)])  # 4, 8, 12, 16, 20
    total_score = (
        jnp.sum(jax.vmap(calculate_hand_value, in_axes=(0, None))(rows, consts))
        + jnp.sum(jax.vmap(calculate_hand_value, in_axes=(0, None))(cols, consts))
        + calculate_hand_value(diagonal1, consts)
        + calculate_hand_value(diagonal2, consts)
    )
    # TODO REVIEW correct calculation and jax.vmap (✿◠‿◠)
    return total_score


@jax.jit
def draw_next_card(state, dealt_cards, consts):
    """Draws the next card from the staple"""
    next_card = jax.lax.cond(
        dealt_cards < consts.NUM_CARDS_IN_DECK,
        lambda s: state.staple[dealt_cards],
        lambda s: jnp.array(0, dtype=jnp.int32),
        state,
    )
    return next_card


@jax.jit
def step_controls(state: CasinoPokerSolitaireState, action: chex.Array, consts: CasinoPokerSolitaireConstants) -> CasinoPokerSolitaireState:
    """ Handles the controls in state 1: move cursor left/right, place card """
    
    ## move cursor left/right
    cursor_x = (
        state.cursor_pos_x
        + jnp.where(action == Action.LEFT, -1, 0)  # add or subtract 1 based on action
        + jnp.where(action == Action.RIGHT, 1, 0)
    )
    # add or subtract to cursor_pos_y based on over/underflow of x, no wrapping
    cursor_y = state.cursor_pos_y + (cursor_x // 5)
    # restrict y within 0-4
    cursor_pos_y = jnp.clip(cursor_y, 0, 4)
    # wrap around x within 0-4 when inside y bounds
    cursor_pos_x = jnp.where(
        (cursor_y >= 0) & (cursor_y <= 4), cursor_x % 5, state.cursor_pos_x
    )
    # no_action_out_of_board = jnp.where((cursor_y < 0) | (cursor_y > 4), 1, 0)

    ## place card
    board = state.board
    current_card = state.current_card
    can_place = (
        (action == Action.FIRE)
        & (current_card != 0)
        & (board[cursor_pos_y * 5 + cursor_pos_x] == 0)
    )
    sound_misplace = (
        (action == Action.FIRE)
        & (current_card != 0)
        & (board[cursor_pos_y * 5 + cursor_pos_x] != -1)
    )  # play error sound

    board, current_card, dealt_cards = jax.lax.cond(
        can_place,
        lambda _: (
            board.at[cursor_pos_y * 5 + cursor_pos_x].set(current_card),
            draw_next_card(state, state.dealt_cards + 1, consts),
            state.dealt_cards + 1,
        ),
        lambda _: (board, current_card, state.dealt_cards),
        operand=None,
    )

    # if 24 cards are placed, move to final scoring state
    state = jax.lax.cond(
        (dealt_cards >= 24) & (state.state_counter == 0),
        lambda s: s._replace(state_counter=1),
        lambda s: s,
        state,
    )

    return state._replace(
        cursor_pos_x=cursor_pos_x,
        cursor_pos_y=cursor_pos_y,
        board=board,
        current_card=current_card,
        dealt_cards=dealt_cards,
    )


@jax.jit
def step_finish_game(state: CasinoPokerSolitaireState, action: chex.Array, consts: CasinoPokerSolitaireConstants) -> CasinoPokerSolitaireState:
    """ Handles the controls in state 2: place remaining card, finish game """
    board = state.board
    current_card = state.current_card

    # place last card in last free spot
    # search free spot in matrix
    # set the current_card at the first position where board == 0
    first_zero_idx = jnp.argmax(board == 0)
    board, dealt_cards = jax.lax.cond(
        (state.dealt_cards == 24) & (state.state_counter == 1),
        lambda _: (board.at[first_zero_idx].set(current_card), state.dealt_cards + 1),
        lambda _: (board, state.dealt_cards),
        operand=None,
    )
    current_card = jnp.array(0, dtype=jnp.int32)
    add_score = calculate_total_score(board, consts)
    cursor_pos = jnp.array([-1, -1])  # hide cursor
    state_counter = 2  # game over

    state = state._replace(
        board=board,
        current_card=current_card,
        state_counter=state_counter,
        player_score=state.player_score + add_score,
        player_round_score=add_score,
        cursor_pos_x=cursor_pos[0],
        cursor_pos_y=cursor_pos[1],
    )
    return state


@jax.jit
def step_end_game(state: CasinoPokerSolitaireState, action: chex.Array, consts: CasinoPokerSolitaireConstants) -> CasinoPokerSolitaireState:
    """ Additional end game state to wait for reset """
    state = jax.lax.cond(
        action == Action.FIRE, lambda s: s._replace(state_counter=3), lambda s: s, state
    )

    return state

class JaxCasinoPokerSolitaire(JaxEnvironment[CasinoPokerSolitaireState, CasinoPokerSolitaireObservation, CasinoPokerSolitaireInfo, CasinoPokerSolitaireConstants]):
    def __init__(self, consts: CasinoPokerSolitaireConstants = None, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE]
        self.obs_size = 5
        self.renderer = CasinoRenderer()
        self.consts = consts or CasinoPokerSolitaireConstants()

    def reset(self, key=None) -> Tuple[CasinoPokerSolitaireObservation, CasinoPokerSolitaireState]:
        # Resets the game state to the initial state, reset score, sample cards
        key1, subkey1 = jax.random.split(key)
        deck = jnp.arange(1, self.consts.NUM_CARDS_IN_DECK + 1, dtype=jnp.int32)
        deck = jax.random.permutation(subkey1, deck)
        state = CasinoPokerSolitaireState(
            key=key if key is not None else jax.random.PRNGKey(0),
            step_counter=jnp.array(0, dtype=jnp.int32),
            state_counter=jnp.array(0, dtype=jnp.int32),
            player_score=self.consts.INITIAL_PLAYER_SCORE,
            player_round_score=jnp.array(0, dtype=jnp.int32),
            staple=deck,
            current_card=jnp.array(deck[0], dtype=jnp.int32),
            dealt_cards=jnp.array(0, dtype=jnp.int32),
            board=jnp.full(
                (25,), 0, dtype=jnp.int32
            ),  # -1 = empty, 0 = no card, 1-52 = card
            cursor_pos_x=jnp.array(1, dtype=jnp.int32),
            cursor_pos_y=jnp.array(
                3, dtype=jnp.int32
            ),  # default pos from the original game
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CasinoPokerSolitaireState, action: chex.Array) -> Tuple[CasinoPokerSolitaireObservation, CasinoPokerSolitaireState, float, bool, CasinoPokerSolitaireInfo]:
        # Steps:
        # move cursor left/right
        # OR execute placement if possible, place card, uncover next card from staple
        # check for end of game -> place remaining card -> calculate final score
        # States:
        # 0: move or place card
        # 1: place last card, calculate final score
        # 2: game over, wait for reset

        previous_state = state
        state = state._replace(
            step_counter=state.step_counter + 1
        )  # increment step counter
        # state machine
        state = jax.lax.cond(
            previous_state.state_counter == 2,
            lambda s, a: step_end_game(s, a, self.consts),
            lambda s, a: s,
            state,
            action,
        )
        state = jax.lax.cond(
            previous_state.state_counter == 1,
            lambda s, a: step_finish_game(s, a, self.consts),
            lambda s, a: s,
            state,
            action,
        )
        state = jax.lax.cond(
            previous_state.state_counter == 0,
            lambda s, a: step_controls(s, a, self.consts),
            lambda s, a: s,
            state,
            action,
        )

        # get reward
        reward = self._get_reward(previous_state, state)
        all_rewards = self._get_all_reward(previous_state, state)
        # get observation
        obs = self._get_observation(state)
        # get info
        info = self._get_info(state, all_rewards)
        # get done
        done = self._get_done(state)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: CasinoPokerSolitaireState, state: CasinoPokerSolitaireState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: CasinoPokerSolitaireState, state: CasinoPokerSolitaireState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CasinoPokerSolitaireState) -> jnp.ndarray:
        card_matrix = jnp.vstack(
            (jnp.array([[0, 0, state.current_card, 0, 0]]), state.board.reshape((5, 5)))
        ).astype(jnp.int32)
        width = self.consts.WIDTH
        height = self.consts.HEIGHT
        player_score = state.player_score
        player_main_bet = jnp.where(
            state.player_round_score == 0, -1, state.player_round_score
        )
        cursor_pos = jnp.array([state.cursor_pos_x, state.cursor_pos_y + 1])

        return self.renderer.render(
            card_matrix,
            player_score,
            width,
            height,
            player_main_bet=player_main_bet,
            blinking_card=cursor_pos,
        )

    def action_space(self) -> Discrete:
        return Discrete(len(self.action_set)) # 4

    def image_space(self) -> Box:
        return Box(
            0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8
        )

    def observation_space(self) -> Dict:
        return Dict(
            {
                "player_score": Box(0, 9999, (), jnp.int32),
                "player_round_score": Box(0, 9999, (), jnp.int32),
                "current_card": Box(0, 52, (), jnp.int32),
                "board": Box(-1, 52, (25,), jnp.int32),
                "cursor_pos_x": Box(0, 4, (), jnp.int32),
                "cursor_pos_y": Box(0, 4, (), jnp.int32),
            }
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CasinoPokerSolitaireObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                obs.player_score.flatten(),
                obs.player_round_score.flatten(),
                obs.current_card.flatten(),
                obs.board.flatten(),
                obs.cursor_pos_x.flatten(),
                obs.cursor_pos_y.flatten(),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: CasinoPokerSolitaireState):
        return CasinoPokerSolitaireObservation(
            player_score=state.player_score.astype(jnp.int32),
            player_round_score=state.player_round_score.astype(jnp.int32),
            current_card=state.current_card.astype(jnp.int32),
            board=state.board.astype(jnp.int32),
            cursor_pos_x=state.cursor_pos_x.astype(jnp.int32),
            cursor_pos_y=state.cursor_pos_y.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CasinoPokerSolitaireState, all_rewards: jnp.ndarray = None) -> CasinoPokerSolitaireInfo:
        return CasinoPokerSolitaireInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: CasinoPokerSolitaireState) -> bool:
        # Board voll
        return state.state_counter == 3