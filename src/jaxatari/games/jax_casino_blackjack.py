#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX Casino Blackjack
#
# Simulates the Atari Casino Blackjack game
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember

from typing import NamedTuple, Tuple
from functools import partial

import os
import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.spaces import Space, Discrete, Box, Dict
from jax_casino_renderer import CasinoRenderer


class CasinoBlackjackConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    INITIAL_PLAYER_SCORE = jnp.array(1000).astype(jnp.int32)
    INITIAL_PLAYER_MAIN_BET = jnp.array(20).astype(jnp.int32)
    CARD_VALUES = jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0])
    CARD_SHUFFLING_RULE = 0
    """ Determines when the cards are being shuffled, 0 = after every round, 1 = after drawing 34 cards """
    ALLOW_SPLITTING = 1
    """ Determines if splitting is allowed (-> when splitting is not allowed additionally the player can take only 5 cards): 0 = splitting is not allowed, 1 = splitting is allowed """
    PLAYING_RULE = 0
    """
    Determines the playing rules, 0 = Casino Rules 1, 1 = Casino Rules 2
    Casino 1 Rules:
    Computer dealer must stay on  hard 17 (Hard means, that any combination of cards is used except an Ace worth 11 points).
    Casino 2 Rules:
    Computer dealer must stay on 17 or more points
    """

class CasinoBlackjackState(NamedTuple):
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

    player_score: chex.Numeric
    """ The score of the player """
    player_main_bet: chex.Numeric
    """ The bet of the player in this round in his main hand (between 1 and 25) """
    player_split_bet: chex.Numeric
    """ The bet of the player in this round in his split hand (between 1 and 25) """
    player_main_action: chex.Numeric
    """ Tells whether hit, stay, double or split is selected on the main hand: 0 = stay, 1 = double, 2 = hit, 3 = split """
    player_split_action: chex.Numeric
    """ Tells whether hit, stay or double is selected on the split hand: 0 = stay, 1 = double, 2 = hit """
    insurance_action: chex.Numeric
    """ Tells whether insurance is selected or not: 0 = not selected (Pass), 1 = insurance selected """
    cards_player_main: chex.Array
    """ An array which includes all the cards of the player (main hand) """
    cards_player_split: chex.Array
    """ An array which includes all cards of the player in the second hand, when splitting """
    cards_player_main_counter: chex.Numeric
    """ The counter for the index which is free in the main hand """
    cards_player_split_counter: chex.Numeric
    """ The counter for the index which is free in the split hand """
    cards_dealer: chex.Array
    """ An array which includes all the of the dealer (same format as player) """
    cards_dealer_counter: chex.Numeric
    """ The counter for the index which is free """
    step_counter: chex.Numeric
    """ The counter for the current step """
    state_counter: chex.Numeric
    """ A counter used for counting the state in the step Method """
    cards_permutation: chex.Array
    """ An array which contains the permutation of the cards which can be drawn (size: 52) """
    card_permutation_counter: chex.Numeric
    """ A counter which tells at which position of the index the next number can be taken """
    key: jax.random.PRNGKey
    """ The key for generating a new key """
    subkey: jax.random.PRNGKey
    """ A subkey for generating random numbers """
    last_round_main: chex.Numeric
    """ Indicates whether the last round on the main hand was won(1) lost(2), draw(3), blackjack(4), bust(5) or there was no last round (0) """
    last_round_split: chex.Numeric
    """ Indicates whether the last round on the split hand was won(1) lost(2), draw(3), blackjack(4), bust(5) or there was no last round (0) """
    is_splitting_selected: chex.Numeric
    """ Indicates whether splitting is selected """
    current_hand: chex.Numeric
    """ Tells whether the main hand or the split hand is selected: 0 = main hand, 1 = split hand """

class CasinoBlackjackObservation(NamedTuple):
    player_score: jnp.ndarray
    """ The score of the player """
    player_main_bet: jnp.ndarray
    """ The bet of the player for his main hand """
    player_split_bet: jnp.ndarray
    """ The bet of the player for his split hand """
    player_current_card_sum_main: jnp.ndarray
    """ The current card sum of the cards in the players main hand """
    player_current_card_sum_split: jnp.ndarray
    """ The current card sum of the cards in the players split hand """
    dealer_current_card_sum: jnp.ndarray
    """ The current card sum of the dealers cards """
    label_main: jnp.ndarray
    """ A label showing the selected player action, the insurance action or the outcome of the last round regarding the players main hand """
    label_split: jnp.ndarray
    """ A label showing the selected player action, the insurance action or the outcome of the last round regarding the players split hand """
    char: jnp.ndarray
    """ Shows an i when the insurance action can be selected, and a question mark, when the bet can be selected """

class CasinoBlackjackInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: jnp.ndarray

@jax.jit
def calculate_card_value_sum(hand: chex.Array, consts):
    """ Calculates the card value of all cards in hand """

    def sum_cards(i, param):
        """ Adds the current card value accept aces to the sum and updates the number of aces """
        csum, number_of_A = param
        value = hand[i]

        # Checks the value for the current card and adds it to csum
        # Aces are ignored
        csum = jnp.where(value != 0, csum + consts.CARD_VALUES[(value - 1) % 13], csum)

        # Checks if the current card is an ace
        number_of_A = jnp.where(jnp.logical_and(value != 0, value % 13 == 0), number_of_A + 1, number_of_A)

        return csum, number_of_A

    # Loops over all cards and returns the sum of their values without the aces and the number of aces
    card_sum, number_of_aces = jax.lax.fori_loop(
        init_val=(0, 0),
        lower=0,
        upper=len(hand),
        body_fun=sum_cards
    )

    def calculate_value_for_A(params):
        """ Decides which value an Ace gets (1 or 11) """
        csum, number_of_A = params

        number_of_A = number_of_A - 1
        csum = csum + number_of_A
        csum = jnp.where(csum > 10, csum + 1, csum + 11)
        return csum

    # Calculates the value for the Aces if there is more than one Ace
    card_sum = jax.lax.cond(
        pred=number_of_aces > 0,
        true_fun=calculate_value_for_A,
        false_fun=lambda o: o[0],
        operand=(card_sum, number_of_aces)
    ).astype(jnp.int32)

    return card_sum

@jax.jit
def select_bet_value(state_action_tuple) -> CasinoBlackjackState:
    """ Step 1: Select bet value with Up/Down and confirm with Fire (and reset displayed cards) """
    state, action, consts = state_action_tuple

    new_player_main_bet = jnp.select(
        condlist=[
            jnp.logical_and(action == Action.UP, state.player_main_bet + 20 <= state.player_score),
            jnp.logical_and(action == Action.UP, state.player_main_bet + 20 > state.player_score),
            action == Action.DOWN
        ],
        choicelist=[
            # If Up is pressed, increase bet by 20
            jnp.minimum(state.player_main_bet + 20, 200),
            # If Up is pressed
            state.player_score - state.player_main_bet,
            # If Down is pressed, decrease bet by 1
            jnp.maximum(state.player_main_bet - 20, 20)
        ],
        # action == Noop or action == Fire
        default=state.player_main_bet
    )

    # Set the counter for the state to 2 if Fire is pressed, otherwise keep it as is
    new_state_counter, new_cards_player_main, new_cards_player_main_counter, new_cards_player_split, new_cards_player_split_counter, new_cards_dealer, new_cards_dealer_counter = jax.lax.cond(
        pred=action == Action.FIRE,
        true_fun=lambda op: (jnp.array(2, dtype=jnp.int32),
                            jnp.zeros(10, dtype=jnp.int32),
                            jnp.array(0, dtype=jnp.int32),
                            jnp.zeros(10, dtype=jnp.int32),
                            jnp.array(0, dtype=jnp.int32),
                            jnp.zeros(10, dtype=jnp.int32),
                            jnp.array(0, dtype=jnp.int32)),
        false_fun=lambda op: (op[0].state_counter, op[0].cards_player_main, op[0].cards_player_main_counter, op[0].cards_player_split, op[0].cards_player_split_counter, op[0].cards_dealer, op[0].cards_dealer_counter),
        operand=(state, consts)
    )

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=new_player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=new_cards_player_main,
        cards_player_main_counter=new_cards_player_main_counter,
        cards_player_split=new_cards_player_split,
        cards_player_split_counter=new_cards_player_split_counter,
        cards_dealer=new_cards_dealer,
        cards_dealer_counter=new_cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter == 2),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=jnp.where(action == Action.FIRE, 0, state.last_round_main),
        last_round_split=jnp.where(action == Action.FIRE, 0, state.last_round_split),
        is_splitting_selected=state.is_splitting_selected,
        current_hand=state.current_hand
    )

    return jax.lax.cond(
        pred=new_state.state_counter == 2,
        true_fun=draw_initial_cards,
        false_fun=lambda o: o[0],
        operand=(new_state, action, consts)
    )

@jax.jit
def draw_initial_cards(state_action_tuple) -> CasinoBlackjackState:
    """ Step 2: Sample card for player, dealer, player and dealer (last card for dealer is hidden) """
    state, action, consts = state_action_tuple

    # Draw cards from the deck
    permutation_counter = state.card_permutation_counter
    card_player1 = state.cards_permutation[permutation_counter]
    card_dealer1 = state.cards_permutation[permutation_counter + 1]
    card_player2 = state.cards_permutation[permutation_counter + 2]
    card_dealer2 = state.cards_permutation[permutation_counter + 3]
    # Increment the counter of the permutation array
    new_card_permutation_counter = jnp.array(permutation_counter + 4).astype(jnp.int32)

    # Give out the cards to the player
    new_cards_player_main = state.cards_player_main.at[state.cards_player_main_counter].set(card_player1)
    new_cards_player_main = new_cards_player_main.at[state.cards_player_main_counter + 1].set(card_player2)
    # Increment the cards_player_main_counter
    new_cards_player_main_counter = jnp.array(state.cards_player_main_counter + 2).astype(jnp.int32)

    # Give out the cards to the dealer
    new_cards_dealer = state.cards_dealer.at[state.cards_dealer_counter].set(card_dealer1)
    new_cards_dealer = new_cards_dealer.at[state.cards_dealer_counter + 1].set(card_dealer2)
    new_cards_dealer_counter = jnp.array(state.cards_dealer_counter + 2).astype(jnp.int32)

    # When dealer has an ace as first card, then go to insurance otherwise check for Blackjack and go to check_winner or when no one has a Blackjack continue with the game
    new_state_counter = jnp.select(
        condlist=[
            consts.CARD_VALUES[(card_dealer1 - 1) % 13] == 0,
            jnp.logical_or(calculate_card_value_sum(new_cards_dealer, consts) == 21, calculate_card_value_sum(new_cards_player_main, consts) == 21)
        ],
        choicelist=[
            jnp.array(3, dtype=jnp.int32),
            jnp.array(12, dtype=jnp.int32)
        ],
        default=jnp.array(4, dtype=jnp.int32),
    )

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=new_cards_player_main,
        cards_player_main_counter=new_cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=new_cards_dealer,
        cards_dealer_counter=new_cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter != 12),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=state.current_hand
    )

    return jax.lax.switch(
        index=new_state_counter - 3 - 7 * (new_state_counter == 12),
        branches=[
            # Step 3
            lambda op: op[0],
            # Step 4
            lambda op: op[0],
            # Step 12
            check_winner
        ],
        operand=(new_state, action, consts),
    )

@jax.jit
def select_insurance_action(state_action_tuple) -> CasinoBlackjackState:
    """ Step 3: Select Insurance or Pass with Up/Down and confirm with Fire """
    state, action, consts = state_action_tuple

    insurance_bet = jnp.floor(state.player_main_bet / 2).astype(jnp.int32)

    new_insurance_action = jax.lax.switch(
        index=state.insurance_action,
        branches=[
            # Insurance action 0
            lambda a: 0 + 1 * (a == Action.UP),
            # Insurance action 1
            lambda a: 1 - 1 * (a == Action.DOWN)
        ],
        operand=action
    )

    has_dealer_blackjack = calculate_card_value_sum(state.cards_dealer, consts) == 21

    # noinspection PyTypeChecker
    new_player_score = jnp.select(
        condlist=[
            jnp.logical_and(action == Action.FIRE, state.insurance_action == 0),
            jnp.logical_and(action == Action.FIRE, state.insurance_action == 1)
        ],
        choicelist=[
            state.player_score,
            jnp.where(has_dealer_blackjack, state.player_score + state.player_main_bet, state.player_score - insurance_bet)
        ],
        default=state.player_score
    )

    new_state_counter = jnp.select(
        condlist=[
            jnp.logical_and(has_dealer_blackjack, action == Action.FIRE),
            action == Action.FIRE
        ],
        choicelist=[
            jnp.array(12, dtype=jnp.int32),
            jnp.array(4, dtype=jnp.int32)
        ],
        default=state.state_counter
    )

    new_state = CasinoBlackjackState(
        player_score=new_player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=new_insurance_action,
        cards_player_main=state.cards_player_main,
        cards_player_main_counter=state.cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter != 12),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=state.current_hand
    )

    return jax.lax.switch(
        index=new_state_counter - 3 - 7 * (new_state_counter == 12),
        branches=[
            # Step 3
            lambda op: op[0],
            # Step 4
            lambda op: op[0],
            # Step 12
            check_winner
        ],
        operand=(new_state, action, consts)
    )

@jax.jit
def select_next_action(state_action_tuple) -> CasinoBlackjackState:
    """ Step 4: Select hit, stay, double or split with Up/Down and confirm with Fire """
    state, action, consts = state_action_tuple

    # Checks if double cannot be executed
    # The player has enough points left when loosing with double selected
    has_enough_points_to_double = jnp.where(state.current_hand == 0, state.player_main_bet, state.player_split_bet) * 2 + jnp.where(state.current_hand == 0, state.player_split_bet, state.player_main_bet) > state.player_score
    # The player can only select double before the first hit
    has_selected_hit_before = jnp.where(state.current_hand == 0, state.cards_player_main_counter, state.cards_player_split_counter) > 2
    # The condition to block selecting double
    block_double = jnp.logical_or(has_enough_points_to_double, has_selected_hit_before)

    # Check if splitting is possible
    # It is the first round and the first and the second card of the player have the same value
    has_same_value_cards = jnp.logical_and(consts.CARD_VALUES[(state.cards_player_main[0] - 1) % 13] == consts.CARD_VALUES[(state.cards_player_main[1] - 1) % 13], state.cards_player_main_counter == 2)
    # The game mode is selected in which splitting is allowed
    splitting_rule_condition = consts.ALLOW_SPLITTING == 1
    # The player has enough points to take a second hand with the same bet amount
    has_enough_points_to_split = state.player_main_bet * 2 <= state.player_score
    # Check if splitting was hit before
    has_not_split_already = state.is_splitting_selected == 0
    # The condition to allow selecting split
    splitting_condition = jnp.logical_and(has_not_split_already, jnp.logical_and(has_enough_points_to_split, jnp.logical_and(has_same_value_cards, splitting_rule_condition)))
    is_splitting_possible = jnp.where(splitting_condition, 1, 0)

    player_action = jnp.where(state.current_hand == 0, state.player_main_action, state.player_split_action)

    # Select player action
    new_player_action = jax.lax.cond(
        pred=jnp.logical_or(action == Action.UP, action == Action.DOWN),
        true_fun=lambda oper: jax.lax.switch(
            index=oper[0],
            branches=[
                # Stay: 0
                lambda op: jax.lax.cond(
                    pred=op[1] == Action.UP,
                    true_fun=lambda o: jnp.where(o[2], o[0] + 2, o[0] + 1),
                    false_fun=lambda o: o[0],
                    operand=op
                ),
                # Double: 1
                lambda op: jax.lax.cond(
                    pred=op[1] == Action.UP,
                    true_fun=lambda o: o[0] + 1,
                    false_fun=lambda o: o[0] - 1,
                    operand=op
                ),
                # Hit: 2
                lambda op: jax.lax.cond(
                    pred=op[1] == Action.UP,
                    true_fun=lambda o: jnp.where(o[3] == 1, o[0] + 1, o[0]),
                    false_fun=lambda o: jnp.where(o[2], o[0] - 2, o[0] - 1),
                    operand=op
                ),
                # Split: 3
                lambda op: jax.lax.cond(
                    pred=op[1] == Action.UP,
                    true_fun=lambda o: o[0],
                    false_fun=lambda o: o[0] - 1,
                    operand=op
                )
            ],
            operand=oper
        ),
        false_fun=lambda oper: oper[0],
        operand=(player_action, action, block_double, is_splitting_possible)
    )

    new_state_counter = jnp.select(
        condlist=[
            state.cards_player_main_counter >= jnp.where(consts.ALLOW_SPLITTING == 0, 5, 10),
            jnp.logical_and(action == Action.FIRE, player_action == 0),
            jnp.logical_and(action == Action.FIRE, player_action == 1),
            jnp.logical_and(action == Action.FIRE, player_action == 2),
            jnp.logical_and(action == Action.FIRE, player_action == 3)
        ],
        choicelist=[
            jnp.array(12, dtype=jnp.int32),
            jnp.array(6, dtype=jnp.int32),
            jnp.array(7, dtype=jnp.int32),
            jnp.array(5, dtype=jnp.int32),
            jnp.array(8, dtype=jnp.int32),
        ],
        default=state.state_counter
    )

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=jnp.where(state.current_hand == 0, new_player_action, jnp.where(state.player_main_action == 3, 0, state.player_main_action)),
        player_split_action=jnp.where(state.current_hand == 1, new_player_action, state.player_split_action),
        insurance_action=state.insurance_action,
        cards_player_main=state.cards_player_main,
        cards_player_main_counter=state.cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter == 4),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=state.current_hand
    )

    return jax.lax.switch(
        index=new_state_counter - 4 - 4 * (new_state_counter == 12),
        branches=[
            # Step 4
            lambda op: op[0],
            # Step 5
            execute_hit,
            # Step 6
            execute_stay,
            # Step 7
            execute_double,
            # Step 8
            execute_split,
            # Step 12
            check_winner
        ],
        operand=(new_state, action, consts)
    )

@jax.jit
def execute_hit(state_action_tuple) -> CasinoBlackjackState:
    """ Step 5: Execute hit """
    state, action, consts = state_action_tuple

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)
    # Change the value at index cards_player_counter to the value of card
    new_cards_player_main = jnp.where(state.current_hand == 0, state.cards_player_main.at[state.cards_player_main_counter].set(card), state.cards_player_main)
    new_cards_player_split = jnp.where(state.current_hand == 1, state.cards_player_split.at[state.cards_player_split_counter].set(card), state.cards_player_split)
    # Increment the cards_player_counter
    new_cards_player_main_counter = jnp.where(state.current_hand == 0, state.cards_player_main_counter + 1, state.cards_player_main_counter)
    new_cards_player_split_counter = jnp.where(state.current_hand == 1, state.cards_player_split_counter + 1, state.cards_player_split_counter)

    # Change next step to state 9
    new_state_counter = jnp.array(9, dtype=jnp.int32)

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=new_cards_player_main,
        cards_player_main_counter=new_cards_player_main_counter,
        cards_player_split=new_cards_player_split,
        cards_player_split_counter=new_cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=state.current_hand
    )

    return check_player_bust((new_state, action, consts))

@jax.jit
def execute_stay(state_action_tuple) -> CasinoBlackjackState:
    """ Step 6: Execute stay """
    state, action, consts = state_action_tuple

    # change the next step to state 10 or to 4 if current hand is split hand
    new_state_counter = jnp.where(state.current_hand == 0, 10, 4)

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=state.cards_player_main,
        cards_player_main_counter=state.cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=0
    )
    return jax.lax.cond(
        pred=state.current_hand == 0,
        true_fun=check_dealer_draw,
        false_fun=lambda op: op[0],
        operand=(new_state, action, consts)
    )

@jax.jit
def execute_double(state_action_tuple) -> CasinoBlackjackState:
    """ Step 7: Execute double """
    state, action, consts = state_action_tuple

    # Double the bet
    new_player_main_bet = jnp.where(state.current_hand == 0, state.player_main_bet * 2, state.player_main_bet)
    new_player_split_bet = jnp.where(state.current_hand == 1, state.player_split_bet * 2, state.player_split_bet)

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)

    # Change the value at index cards_player_counter to the value of card
    new_cards_player_main = jnp.where(state.current_hand == 0, state.cards_player_main.at[state.cards_player_main_counter].set(card), state.cards_player_main)
    new_cards_player_split = jnp.where(state.current_hand == 1, state.cards_player_split.at[state.cards_player_split_counter].set(card), state.cards_player_split)
    # Increment the cards_player_counter
    new_cards_player_main_counter = jnp.where(state.current_hand == 0, state.cards_player_main_counter + 1, state.cards_player_main_counter)
    new_cards_player_split_counter = jnp.where(state.current_hand == 1, state.cards_player_split_counter + 1, state.cards_player_split_counter)

    # check hand, change the next step
    new_state_counter = jnp.select(
        condlist=[
            state.current_hand == 1,
            calculate_card_value_sum(new_cards_player_main, consts) > 21
        ],
        choicelist=[
            4,
            12
        ],
        default=10,
    )

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=new_player_main_bet,
        player_split_bet=new_player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=new_cards_player_main,
        cards_player_main_counter=new_cards_player_main_counter,
        cards_player_split=new_cards_player_split,
        cards_player_split_counter=new_cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=0
    )

    return jax.lax.switch(
        index=jnp.array(new_state_counter / 2, dtype=jnp.int32) - 2 - 2 * (new_state_counter != 2),
        branches=[
            # Step 4
            lambda op: op[0],
            # Step 10
            check_dealer_draw,
            # Step 12
            check_winner
        ],
        operand=(new_state, action, consts),
    )

@jax.jit
def execute_split(state_action_tuple) -> CasinoBlackjackState:
    """ Step 8: Execute split """
    state, action, consts = state_action_tuple

    card_1 = state.cards_permutation[state.card_permutation_counter]
    card_2 = state.cards_permutation[state.card_permutation_counter + 1]

    new_permutation_counter = state.card_permutation_counter + 2

    # Second card comes to split hand
    new_cards_player_split = state.cards_player_split.at[0].set(state.cards_player_main[1])
    new_cards_player_split = new_cards_player_split.at[1].set(card_1)
    new_cards_player_split_counter = jnp.array(2, dtype=jnp.int32)

    # First card stays in main hand
    new_cards_player_main = state.cards_player_main.at[1].set(card_2)
    new_cards_player_main_counter = jnp.array(2, dtype=jnp.int32)

    is_splitting_selected = jnp.array(1, dtype=jnp.int32)

    # Split hand will be selected
    new_current_hand = jnp.array(1, dtype=jnp.int32)

    # Set the player bet on the split hand
    new_player_split_bet = state.player_main_bet

    # Change next step to state 9
    new_state_counter = jnp.array(4, dtype=jnp.int32)

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=new_player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=new_cards_player_main,
        cards_player_main_counter=new_cards_player_main_counter,
        cards_player_split=new_cards_player_split,
        cards_player_split_counter=new_cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=is_splitting_selected,
        current_hand=new_current_hand
    )

    return new_state

@jax.jit
def check_player_bust(state_action_tuple) -> CasinoBlackjackState:
    """ Step 9: Check if player has 21 or more """
    state, action, consts = state_action_tuple

    player_cards_sum = calculate_card_value_sum(jnp.where(state.current_hand == 0, state.cards_player_main, state.cards_player_split), consts)

    new_state_counter = jnp.select(
        condlist=[
            jnp.logical_and(player_cards_sum > 21, state.current_hand == 0),
            jnp.logical_and(player_cards_sum == 21, state.current_hand == 0) ,
            player_cards_sum < 21
        ],
        choicelist=[
            jnp.array(12, dtype=jnp.int32),
            jnp.array(10, dtype=jnp.int32),
            jnp.array(4, dtype=jnp.int32)
        ],
        # player_card_sum < 21 -> player can take more cards or player is switching to main hand
        default=jnp.array(4, dtype=jnp.int32)
    )

    new_current_hand = jnp.where(jnp.logical_and(player_cards_sum >= 21, state.current_hand == 1), 0, state.current_hand)

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=state.cards_player_main,
        cards_player_main_counter=state.cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter == 4),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=new_current_hand
    )

    return jax.lax.switch(
        index=new_state_counter - 10 + 8 * (new_state_counter == 4) - 1 * (new_state_counter == 12),
        branches=[
            # Step 10
            check_dealer_draw,
            # Step 12
            check_winner,
            # Step 4
            lambda op: op[0]
        ],
        operand=(new_state, action, consts)
    )

@jax.jit
def check_dealer_draw(state_action_tuple) -> CasinoBlackjackState:
    """ Step 10: Check if dealer should take a card """
    state, action, consts = state_action_tuple

    card_sum = calculate_card_value_sum(state.cards_dealer, consts)
    condition = jnp.array([
        jnp.logical_or(card_sum > 17, jnp.logical_and(card_sum >= 17, jnp.any(jnp.isin(state.cards_dealer, jnp.array([25, 26, 51, 52]))))),
        card_sum > 16
    ], dtype=bool)

    new_state_counter = jax.lax.cond(
        pred=jnp.logical_and(jnp.logical_not(state.cards_dealer_counter == jnp.where(consts.ALLOW_SPLITTING == 0, 5, 10)), jnp.logical_not(condition[consts.PLAYING_RULE])),
        true_fun=lambda: jnp.array(11, dtype=jnp.int32),
        false_fun=lambda: jnp.array(12, dtype=jnp.int32)
    )

    return jax.lax.cond(
        pred=new_state_counter == 12,
        true_fun=check_winner,
        false_fun=execute_dealer_draw,
        operand=(state, action, consts)
    )

@jax.jit
def execute_dealer_draw(state_action_tuple) -> CasinoBlackjackState:
    """ Step 11: Give dealer a card """
    state, action, consts = state_action_tuple

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1, dtype=jnp.int32)
    # Copy cards_dealer and change the value at index cards_dealer_counter to the value of card
    new_cards_dealer = state.cards_dealer.at[state.cards_dealer_counter].set(card)
    # Increment the cards_dealer_counter
    new_cards_dealer_counter = jnp.array(state.cards_dealer_counter + 1, dtype=jnp.int32)
    # Change next step to state 10
    new_state_counter = jnp.array(10, dtype=jnp.int32)

    new_state = CasinoBlackjackState(
        player_score=state.player_score,
        player_main_bet=state.player_main_bet,
        player_split_bet=state.player_split_bet,
        player_main_action=state.player_main_action,
        player_split_action=state.player_split_action,
        insurance_action=state.insurance_action,
        cards_player_main=state.cards_player_main,
        cards_player_main_counter=state.cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=new_cards_dealer,
        cards_dealer_counter=new_cards_dealer_counter,
        step_counter=state.step_counter + 1,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round_main=state.last_round_main,
        last_round_split=state.last_round_split,
        is_splitting_selected=state.is_splitting_selected,
        current_hand=state.current_hand
    )

    return new_state

@jax.jit
def check_winner(state_action_tuple) -> CasinoBlackjackState:
    """ Step 12: Check winner """
    state, action, consts = state_action_tuple

    # Calculate player_score and last_round for main hand and split hand
    player_score, new_last_round_main = calculate_player_score_and_last_round(state, state.player_score, consts, 0)
    player_score, new_last_round_split = jax.lax.cond(
        pred=state.cards_player_split_counter > 0,
        true_fun=lambda op: calculate_player_score_and_last_round(op[0], op[1], op[2], 1),
        false_fun=lambda op: (op[1], 0),
        operand=(state, player_score, consts)
    )

    # Condition for mixing the cards
    condition = jnp.logical_and(state.current_hand == 0, jnp.logical_or(consts.CARD_SHUFFLING_RULE == 0, jnp.logical_and(consts.CARD_SHUFFLING_RULE == 1, state.card_permutation_counter >= 34)))

    # Generates a new permutation of the cards and keys and resets the counter if necessary
    new_cards_permutation, new_card_permutation_counter, new_key, new_subkey = jax.lax.cond(
        pred=condition,
        true_fun=lambda s: (jax.random.permutation(s.subkey, jnp.arange(1, 53)), jnp.array(0, dtype=jnp.int32), *jax.random.split(s.key)),
        false_fun=lambda s: (s.cards_permutation, s.card_permutation_counter, s.key, s.subkey),
        operand=state,
    )

    new_state = CasinoBlackjackState(
        player_score=player_score,
        player_main_bet=jnp.minimum(state.player_main_bet, state.player_score),
        player_split_bet=jnp.array(0, dtype=jnp.int32),
        player_main_action=jnp.array(0, dtype=jnp.int32),
        player_split_action=jnp.array(0, dtype=jnp.int32),
        insurance_action=jnp.array(0, dtype=jnp.int32),
        cards_player_main=state.cards_player_main,
        cards_player_main_counter=state.cards_player_main_counter,
        cards_player_split=state.cards_player_split,
        cards_player_split_counter=state.cards_player_split_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1,
        state_counter=jnp.array(1, dtype=jnp.int32),
        cards_permutation=new_cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=new_key,
        subkey=new_subkey,
        last_round_main=new_last_round_main,
        last_round_split=new_last_round_split,
        is_splitting_selected=jnp.array(0, dtype=jnp.int32),
        current_hand=jnp.array(0, dtype=jnp.int32)
    )

    return new_state

@jax.jit
def calculate_player_score_and_last_round(state, player_score, consts, hand_to_evaluate):
    """ Calculates the new player score and last round """

    bet = jnp.where(hand_to_evaluate == 0, state.player_main_bet, state.player_split_bet)
    hand_player = jnp.where(hand_to_evaluate == 0, calculate_card_value_sum(state.cards_player_main, consts), calculate_card_value_sum(state.cards_player_split, consts))
    hand_dealer = calculate_card_value_sum(state.cards_dealer, consts)
    cards_player_counter = jnp.where(hand_to_evaluate == 0, state.cards_player_main_counter, state.cards_player_split_counter)

    def player_wins1x():
        """ Handles the case where the player wins 1x """
        return (player_score + bet, jnp.array(1, dtype=jnp.int32))

    def player_looses():
        """ Handles the case where the player looses and the dealer wins """
        return (player_score - bet, jnp.array(2, dtype=jnp.int32))

    def handle_draw():
        """ Handles the draw case """
        return (player_score, jnp.array(3, dtype=jnp.int32))

    # -# check win #-#
    # noinspection PyTypeChecker
    new_player_score, new_last_round = jnp.select(
        condlist=[
            hand_player > 21,  # player has more than 21
            jnp.logical_and(jnp.logical_and(hand_player == 21, cards_player_counter == 2), jnp.logical_and(hand_dealer == 21, state.cards_dealer_counter == 2)),  # player and dealer have a Blackjack
            jnp.logical_and(jnp.logical_and(hand_player == 21, cards_player_counter == 2), state.is_splitting_selected == 0),  # only player has a Blackjack
            jnp.logical_or(hand_dealer > 21, cards_player_counter >= jnp.where(consts.ALLOW_SPLITTING == 0, 5, 10)),  # dealer has more than 21 OR  player has 5 or 10 cards without busting
            hand_player > hand_dealer,  # player has more than dealer
            hand_player < hand_dealer,  # player has less than dealer
            hand_player == hand_dealer,  # player and dealer have the same points
        ],
        choicelist=jnp.array([
            (state.player_score - bet, jnp.array(5, dtype=jnp.int32)),  # -> bust: player looses
            handle_draw(),
            (jnp.floor(state.player_score + (bet * 1.5)).astype(jnp.int32), jnp.array(4, dtype=jnp.int32)),  # -> player wins (1.5x)
            player_wins1x(),
            player_wins1x(),
            player_looses(),
            handle_draw()
        ])
    )

    return new_player_score, new_last_round

class JaxCasinoBlackjack(JaxEnvironment[CasinoBlackjackState, CasinoBlackjackObservation, CasinoBlackjackInfo, CasinoBlackjackConstants]):
    def __init__(self, consts: CasinoBlackjackConstants = None, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.FIRE
        ]
        self.obs_size = 11
        self.renderer = CasinoRenderer()
        self.consts = consts or CasinoBlackjackConstants()

    def reset(self, key=jax.random.PRNGKey(int.from_bytes(os.urandom(3), byteorder='big'))) -> Tuple[CasinoBlackjackObservation, CasinoBlackjackState]:
        # Resets the game state to the initial state.
        key1, subkey1 = jax.random.split(key)
        key2, subkey2 = jax.random.split(key1)
        state = CasinoBlackjackState(
            player_score=self.consts.INITIAL_PLAYER_SCORE,
            player_main_bet=self.consts.INITIAL_PLAYER_MAIN_BET,
            player_split_bet=jnp.array(0, dtype=jnp.int32),
            player_main_action=jnp.array(0, dtype=jnp.int32),
            player_split_action=jnp.array(0, dtype=jnp.int32),
            insurance_action=jnp.array(1, dtype=jnp.int32),
            cards_player_main=jnp.zeros(10, dtype=jnp.int32),
            cards_player_main_counter=jnp.array(0, dtype=jnp.int32),
            cards_player_split=jnp.zeros(10, dtype=jnp.int32),
            cards_player_split_counter=jnp.array(0, dtype=jnp.int32),
            cards_dealer=jnp.zeros(10, dtype=jnp.int32),
            cards_dealer_counter=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            state_counter=jnp.array(1, dtype=jnp.int32),
            cards_permutation=jax.random.permutation(subkey1, jnp.arange(1, 53)),
            card_permutation_counter=jnp.array(0, dtype=jnp.int32),
            key=key2,
            subkey=subkey2,
            last_round_main=jnp.array(0, dtype=jnp.int32),
            last_round_split=jnp.array(0, dtype=jnp.int32),
            is_splitting_selected=jnp.array(0, dtype=jnp.int32),
            current_hand=jnp.array(0, dtype=jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CasinoBlackjackState, action: chex.Array) -> Tuple[CasinoBlackjackObservation, CasinoBlackjackState, float, bool, CasinoBlackjackInfo]:
        new_state = jax.lax.switch(
            index=state.state_counter - 1,
            branches=[select_bet_value, draw_initial_cards, select_insurance_action, select_next_action, execute_hit, execute_stay, execute_double, execute_split, check_player_bust, check_dealer_draw, execute_dealer_draw, check_winner],
            operand=(state, action, self.consts)
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: CasinoBlackjackState, state: CasinoBlackjackState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: CasinoBlackjackState, state: CasinoBlackjackState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state)
             for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CasinoBlackjackState) -> jnp.ndarray:
        card_matrix = jnp.array([
            [state.cards_dealer[0], jnp.where(jnp.logical_and(state.state_counter >= 2, state.state_counter <= 8), -1, state.cards_dealer[1]), state.cards_dealer[2], state.cards_dealer[3], state.cards_dealer[4]],
            [jnp.where(self.consts.ALLOW_SPLITTING == 1, state.cards_dealer[5], 0), jnp.where(self.consts.ALLOW_SPLITTING == 1, state.cards_dealer[6], 0), jnp.where(self.consts.ALLOW_SPLITTING == 1, state.cards_dealer[7], 0), jnp.where(self.consts.ALLOW_SPLITTING == 1, state.cards_dealer[8], 0), jnp.where(self.consts.ALLOW_SPLITTING == 1, state.cards_dealer[9], 0),],
            [state.cards_player_main[0], state.cards_player_main[1], state.cards_player_main[2], state.cards_player_main[3], state.cards_player_main[4]],
            [state.cards_player_main[5], state.cards_player_main[6], state.cards_player_main[7], state.cards_player_main[8], state.cards_player_main[9]],
            [state.cards_player_split[0], state.cards_player_split[1], state.cards_player_split[2], state.cards_player_split[3], state.cards_player_split[4]],
            [state.cards_player_split[5], state.cards_player_split[6], state.cards_player_split[7], state.cards_player_split[8], state.cards_player_split[9]]
        ], dtype=jnp.int32)
        width = self.consts.WIDTH
        height = self.consts.HEIGHT
        player_score, player_main_bet, player_split_bet, label_main, label_split, char = self.get_params(state)

        return self.renderer.render(card_matrix, player_score, width, height, char, player_main_bet, player_split_bet, label_main, label_split)

    @partial(jax.jit, static_argnums=(0,))
    def get_params(self, state: CasinoBlackjackState):
        """ Calculates all values from state needed for observation and the renderer """
        player_score = state.player_score.astype(jnp.int32)
        char = jnp.select(
            condlist=[
                state.state_counter == 3,
                state.state_counter == 1
            ],
            choicelist=[
                0,
                1
            ],
            default=-1
        ).astype(jnp.int32)
        player_main_bet = state.player_main_bet.astype(jnp.int32)
        player_split_bet = jnp.where(state.player_split_bet != 0, state.player_split_bet, -1).astype(jnp.int32)
        label_main = jnp.select(
            condlist=[
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.last_round_main == 4, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.last_round_main == 5, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.player_main_action == 1, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.player_main_action == 2, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.insurance_action == 1, state.state_counter == 3)),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.last_round_main == 2, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.insurance_action == 0, state.state_counter == 3)),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.last_round_main == 3, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.player_main_action == 3, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.player_main_action == 0, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.current_hand == 0, jnp.logical_and(state.last_round_main == 1, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32))))
            ],
            # 1 = BJ, 2 = Bust, 3 = Double, 5 = Hit, 6 = Insr, 7 = Lose, 8 = Pass, 9 = Push, 10 = Split, 11 = Stay, 12 = Win
            choicelist=[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
            default=-1
        ).astype(jnp.int32),
        label_split = jnp.select(
            condlist=[
                jnp.logical_and(state.cards_player_split_counter > 0, jnp.logical_and(state.last_round_split == 4, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.cards_player_split_counter > 0, jnp.logical_and(state.last_round_split == 5, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.is_splitting_selected == 1, jnp.logical_and(state.player_split_action == 1, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.is_splitting_selected == 1, jnp.logical_and(state.player_split_action == 2, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.cards_player_split_counter > 0, jnp.logical_and(state.last_round_split == 2, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.cards_player_split_counter > 0, jnp.logical_and(state.last_round_split == 3, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)))),
                jnp.logical_and(state.is_splitting_selected == 1, jnp.logical_and(state.player_split_action == 0, jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)))),
                jnp.logical_and(state.cards_player_split_counter > 0, jnp.logical_and(state.last_round_split == 1, jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32))))
            ],
            # 1 = BJ, 2 = Bust, 3 = Double, 5 = Hit, 7 = Lose, 9 = Push, 11 = Stay, 12 = Win
            choicelist=[1, 2, 3, 5, 7, 9, 11, 12],
            default=-1
        ).astype(jnp.int32)

        return player_score, player_main_bet, player_split_bet, label_main[0], label_split, char

    def action_space(self) -> Discrete:
        return Discrete(len(self.action_set))

    def image_space(self) -> Box:
        return Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> Dict:
        return Dict({
            "player_score": Box(0, 9999, (), jnp.int32),
            "player_main_bet": Box(1, 200, (), jnp.int32),
            "player_split_bet": Box(-1, 200, (), jnp.int32),
            "player_current_card_sum_main": Box(0, 30, (), jnp.int32),
            "player_current_card_sum_split": Box(0, 30, (), jnp.int32),
            "dealer_current_card_sum": Box(0, 30, (), jnp.int32),
            "label_main": Box(-1, 12, (), jnp.int32),
            "label_split": Box(-1, 12, (), jnp.int32),
            "char": Box(-1, 1, (), jnp.int32)
        })

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CasinoBlackjackObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_score.flatten(),
            obs.player_main_bet.flatten(),
            obs.player_split_bet.flatten(),
            obs.player_current_card_sum_main.flatten(),
            obs.player_current_card_sum_split.flatten(),
            obs.dealer_current_card_sum.flatten(),
            obs.label_main.flatten(),
            obs.label_split.flatten(),
            obs.char.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: CasinoBlackjackState):
        player_main_sum = calculate_card_value_sum(state.cards_player_main, self.consts)
        player_split_sum = calculate_card_value_sum(state.cards_player_split, self.consts)

        dealer_sum = jax.lax.cond(
            pred=state.cards_dealer_counter == 2,
            true_fun=lambda s: calculate_card_value_sum(jnp.concatenate([s.cards_dealer[:1], s.cards_dealer[1:]]), self.consts),
            false_fun=lambda s: calculate_card_value_sum(state.cards_dealer, self.consts),
            operand=state
        )

        player_score, player_main_bet, player_split_bet, label_main, label_split, char = self.get_params(state)

        return CasinoBlackjackObservation(
            player_score=player_score,
            player_main_bet=player_main_bet,
            player_split_bet=player_split_bet,
            player_current_card_sum_main=player_main_sum,
            player_current_card_sum_split=player_split_sum,
            dealer_current_card_sum=dealer_sum,
            label_main=label_main,
            label_split=label_split,
            char=char,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CasinoBlackjackState, all_rewards: jnp.ndarray = None) -> CasinoBlackjackInfo:
        return CasinoBlackjackInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: CasinoBlackjackState) -> bool:
        player_has_no_money = state.player_score <= 0
        player_wins = state.player_score >= 10000
        return jnp.logical_or(player_has_no_money, player_wins)