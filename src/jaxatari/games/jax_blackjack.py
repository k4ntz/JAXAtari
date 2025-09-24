#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX Blackjack
#
# Simulates a Blackjack game with casino rules using JAX
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember

import os
from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.spaces import Space, Discrete, Box, Dict


class BlackjackConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    INITIAL_PLAYER_SCORE = jnp.array(200).astype(jnp.int32)
    INITIAL_PLAYER_BET = jnp.array(1).astype(jnp.int32)
    CARD_VALUES = jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0])
    SKIP_FRAMES = 0
    """ Determines whether frames are skipped: 0 = no, 1 = yes """
    CARD_SHUFFLING_RULE = 0
    """ Determines when the cards are being shuffled: 0 = after every round, 1 = after drawing 34 cards """
    PLAYING_RULE = 0
    """ Determines the playing rules, 0 = Casino Rules, 1 = Private Rules 
    
    CASINO BLACK JACK RULES:
    - Computer dealer must hit a soft 17 or less. (Soft means, that an Ace is used as 11 points).
    - Computer dealer must stay on a hard 17. (Hard means, that any combination of cards is used except an Ace worth 11 points).
    - The player gets no points, but he looses also no points if there is a tie.
    - The player is only allowed to hit double before the first hit, and he needs to have 10 or 11 points.
    - A player is allowed four hits.

    PRIVATE BLACK JACK RULES:
    - Computer dealer must stay on 17 or more points.
    - The dealer wins all tie games.
    - The player is only allowed to hit double before the first hit but with any combination of cards.
    - A player wins the game when he hits four times without busting.
    """



class BlackjackState(NamedTuple):
    # Format: [n, n, 0, 0, 0, 0] (Array has a size of 6)
    # 0 -> no card currently
    # n Values: 01 -> B2            14 -> B2            27 -> R2            40 -> R2
    #           02 -> B3            15 -> B3            28 -> R3            41 -> R3
    #           03 -> B4            16 -> B4            29 -> R4            42 -> R4
    #           ...                 ...                 ...                 ...
    #           09 -> B10           22 -> B10           35 -> R10           48 -> R10
    #           10 -> BJ            23 -> BJ            36 -> RJ            49 -> RJ
    #           11 -> BK            24 -> BK            37 -> RK            50 -> RK
    #           12 -> BQ            25 -> BQ            38 -> RQ            51 -> RQ
    #           13 -> BA            26 -> BA            39 -> RA            52 -> RA
    player_score: chex.Numeric
    """ The score of the player """
    player_bet: chex.Numeric
    """ The bet of the player in this round (between 1 and 25) """
    player_action: chex.Numeric
    """ Tells whether hit stay or double is selected: 0 = stay, 1 = double, 2 = hit """
    cards_player: chex.Array
    """ An array which includes all the cards of the player """
    cards_player_counter: chex.Numeric
    """ The counter for the index which is free """
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
    last_round: chex.Numeric
    """ Indicates whether the last round was won (1) lost(2), draw(3), blackjack(4), bust(5) or there was no last round (0) """
    skip_step: chex.Numeric
    """ Contains a target tick count. The game will not continue until the tick count is reached """


class BlackjackObservation(NamedTuple):
    player_score: jnp.ndarray
    """ The player score """
    player_bet: jnp.ndarray
    """ The player bet """
    player_current_card_sum: jnp.ndarray
    """ The current card sum of the player """
    dealer_current_card_sum: jnp.ndarray
    """ The current card sum of the dealer """
    label: jnp.ndarray
    """ A label showing the selected player action or the outcome of the last round regarding the players main hand 
    0 = no label, 1 = BJ, 2 = Bust, 3 = Double, 4 = Hit, 5 = Lose, 6 = Tie, 7 = Stay, 8 = Win
    """


class BlackjackInfo(NamedTuple):
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
    )

    return jnp.array(card_sum).astype(jnp.int32)

@jax.jit
def select_bet_value(state_action_tuple) -> BlackjackState:
    """ Step 1: Select bet value with Up/Down and confirm with Fire (and reset displayed cards) """
    state, action, consts = state_action_tuple

    # Check if the action is Up, Down or Fire
    up = action == Action.UP
    down = action == Action.DOWN
    fire = action == Action.FIRE

    new_player_bet = jnp.select(
        condlist=[
            jnp.logical_and(action == Action.UP, state.player_bet < state.player_score),
            action == Action.DOWN
        ],
        choicelist=[
            # If Up is pressed, increase bet by 1
            jnp.minimum(state.player_bet + 1, 25),
            # If Down is pressed, decrease bet by 1
            jnp.maximum(state.player_bet - 1, 1)
        ],
        # action == Noop or action == Fire
        default=state.player_bet
    )

    # Set the counter for the state to 2 if Fire is pressed, otherwise keep it as is
    new_state_counter, new_cards_player, new_cards_player_counter, new_cards_dealer, new_cards_dealer_counter = jax.lax.cond(
        pred=fire,
        true_fun=lambda _: (2, jnp.array([0, 0, 0, 0, 0, 0]).astype(jnp.int32), jnp.array(0).astype(jnp.int32), jnp.array([0, 0, 0, 0, 0, 0]).astype(jnp.int32), jnp.array(0).astype(jnp.int32)),
        false_fun=lambda op: op,
        operand=(state.state_counter, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter)
    )

    new_skip_state = jnp.where(action != Action.NOOP, state.step_counter + 6, state.step_counter + 1)

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=new_player_bet,
        player_action=state.player_action,
        cards_player=new_cards_player,
        cards_player_counter=new_cards_player_counter,
        cards_dealer=new_cards_dealer,
        cards_dealer_counter=new_cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter != 2),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=new_skip_state
    )

    return jax.lax.cond(
        pred=new_state.state_counter == 2,
        true_fun=draw_initial_cards,
        false_fun=lambda o: o[0],
        operand=(new_state, action, consts)
    )

@jax.jit
def draw_initial_cards(state_action_tuple) -> BlackjackState:
    """ Step 2: Sample card for player, dealer, player and dealer (last card for dealer is hidden) """
    state, action, consts = state_action_tuple

    card = state.cards_permutation[state.card_permutation_counter]
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)

    give_player_condition = state.cards_player_counter == state.cards_dealer_counter

    new_cards_player = jnp.where(give_player_condition, state.cards_player.at[state.cards_player_counter].set(card), state.cards_player)
    new_cards_player_counter = state.cards_player_counter + jnp.where(give_player_condition, 1, 0)

    new_cards_dealer = jnp.where(give_player_condition, state.cards_dealer, state.cards_dealer.at[state.cards_dealer_counter].set(card))
    new_cards_dealer_counter = state.cards_dealer_counter + jnp.where(give_player_condition, 0, 1)

    new_state_counter = jax.lax.cond(
        pred=jnp.logical_and(new_cards_player_counter == 2, new_cards_dealer_counter == 2),
        true_fun=lambda op: jnp.where(
            jnp.logical_or(calculate_card_value_sum(op[0], consts) == 21, calculate_card_value_sum(op[1], consts) == 21),
            jnp.array(10).astype(jnp.int32),
            jnp.array(3).astype(jnp.int32)
        ),
        false_fun=lambda _: 2,
        operand=(new_cards_dealer, new_cards_player)
    )

    new_skip_step = jnp.where(new_state_counter == 2, state.step_counter + 41, state.step_counter + 1)

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=state.player_bet,
        player_action=state.player_action,
        cards_player=new_cards_player,
        cards_player_counter=new_cards_player_counter,
        cards_dealer=new_cards_dealer,
        cards_dealer_counter=new_cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter != 10),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=new_skip_step
    )

    return jax.lax.switch(
        index=new_state_counter - 2 - 6 * (new_state_counter == 10),
        branches=[
            # Step 2
            lambda op: op[0],
            # Step 3
            lambda op: op[0],
            # Step 10
            check_winner
        ],
        operand=(new_state, action, consts)
    )

@jax.jit
def select_next_action(state_action_tuple) -> BlackjackState:
    """ Step 3: Select hit, stay or double with Up/Down and confirm with Fire """
    state, action, consts = state_action_tuple
    player_cards_sum = calculate_card_value_sum(state.cards_player, consts)

    # Conditions
    # The player has enough points left when loosing with double selected
    has_enough_points_to_double = state.player_bet * 2 > state.player_score
    # If Playing Rule 0 is selected, then the player can select double only if he has a card sum of 10 or 11
    playing_rule_condition = jnp.logical_or(jnp.logical_and(jnp.logical_and(player_cards_sum != 10, player_cards_sum != 11), consts.PLAYING_RULE == 0), consts.PLAYING_RULE == 1)
    # The player can only select double before the first hit
    has_selected_hit_before = state.cards_player_counter > 2
    # The condition to block selecting double
    block_double = jnp.logical_or(jnp.logical_or(has_enough_points_to_double, playing_rule_condition), has_selected_hit_before)

    # player action to select: 0 = stay, 1 = double, 2 = hit
    new_player_action = jax.lax.cond(
        pred=jnp.logical_or(jnp.logical_and(action == Action.UP, state.player_action == 2), jnp.logical_and(action == Action.DOWN, state.player_action == 0)),
        true_fun=lambda op: op[0],
        false_fun=lambda op: jax.lax.cond(
            pred=jnp.logical_and(op[0] == (op[1] == Action.DOWN) * 2, op[2]),
            true_fun=lambda o: o[0] + 2 * (o[1] == Action.UP) - 2 * (o[1] == Action.DOWN),
            false_fun=lambda o: o[0] + 1 * (o[1] == Action.UP) - 1 * (o[1] == Action.DOWN),
            operand=(op[0], op[1], op[2])
        ),
        operand=(state.player_action, action, block_double)
    )

    new_state_counter = jnp.select(
        condlist=[
            state.cards_player_counter >= 6,
            jnp.logical_and(action == Action.FIRE, state.player_action == 0),
            jnp.logical_and(action == Action.FIRE, state.player_action == 1),
            jnp.logical_and(action == Action.FIRE, state.player_action == 2)
        ],
        choicelist=[
            jnp.array(10).astype(jnp.int32),
            jnp.array(5).astype(jnp.int32),
            jnp.array(6).astype(jnp.int32),
            jnp.array(4).astype(jnp.int32)
        ],
        default=state.state_counter
    )

    new_skip_step = jnp.where(action != Action.NOOP, state.step_counter + 11, state.step_counter + 1)

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=state.player_bet,
        player_action=new_player_action,
        cards_player=state.cards_player,
        cards_player_counter=state.cards_player_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter == 3),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=new_skip_step
    )

    return jax.lax.switch(
        index=new_state_counter - 3 - 3 * (new_state_counter == 10),
        branches=[
            # Step 3
            lambda op: op[0],
            # Step 4
            execute_hit,
            # Step 5
            execute_stay,
            # Step 6
            execute_double,
            # Step 10
            check_winner
        ],
        operand=(new_state, action, consts)
    )

@jax.jit
def execute_hit(state_action_tuple) -> BlackjackState:
    """ Step 4: Execute hit """
    state, action, consts = state_action_tuple

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)
    # Copy cards_player and change the value at index cards_player_counter to the value of card
    new_cards_player = state.cards_player.at[state.cards_player_counter].set(card)
    # Increment the cards_player_counter
    new_cards_player_counter = jnp.array(state.cards_player_counter + 1).astype(jnp.int32)
    # Change next step to state 7
    new_state_counter = jnp.array(7).astype(jnp.int32)

    new_skip_step = state.step_counter + 41

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=state.player_bet,
        player_action=state.player_action,
        cards_player=new_cards_player,
        cards_player_counter=new_cards_player_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=new_skip_step
    )

    return check_player_bust((new_state, action, consts))

@jax.jit
def execute_stay(state_action_tuple) -> BlackjackState:
    """ Step 5: Execute stay """
    state, action, consts = state_action_tuple

    # change the next step to state 8
    new_state_counter = jnp.array(8).astype(jnp.int32)

    return check_dealer_draw((state, action, consts))

@jax.jit
def execute_double(state_action_tuple) -> BlackjackState:
    """ Step 6: Execute double """
    state, action, consts = state_action_tuple

    # Double the bet
    new_player_bet = jnp.array(state.player_bet * 2).astype(jnp.int32)

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)

    # Give card
    new_cards_player = state.cards_player.at[state.cards_player_counter].set(card)
    # Increment the cards_player_counter
    new_cards_player_counter = jnp.array(state.cards_player_counter + 1).astype(jnp.int32)

    # check hand, change the next step
    new_state_counter = jnp.where(calculate_card_value_sum(new_cards_player, consts) > 21, jnp.array(10).astype(jnp.int32), jnp.array(8).astype(jnp.int32))

    new_skip_step = state.step_counter + 41

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=new_player_bet,
        player_action=state.player_action,
        cards_player=new_cards_player,
        cards_player_counter=new_cards_player_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=new_skip_step
    )

    return jax.lax.cond(
        pred=new_state.state_counter == 8,
        true_fun=check_dealer_draw,
        false_fun=check_winner,
        operand=(new_state, action, consts),
    )

@jax.jit
def check_player_bust(state_action_tuple) -> BlackjackState:
    """ Step 7: Check if player has 21 or more """
    state, action, consts = state_action_tuple

    player_cards_sum = calculate_card_value_sum(state.cards_player, consts)

    new_state_counter = jnp.select(
        condlist=[
            player_cards_sum > 21,
            player_cards_sum == 21,
            player_cards_sum < 21
        ],
        choicelist=[
            jnp.array(10).astype(jnp.int32),
            jnp.array(8).astype(jnp.int32),
            jnp.array(3).astype(jnp.int32)
        ],
        # player_card_sum < 21 -> player can take more cards
        default=jnp.array(3).astype(jnp.int32)
    )

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=state.player_bet,
        player_action=state.player_action,
        cards_player=state.cards_player,
        cards_player_counter=state.cards_player_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1 * (new_state_counter == 3),
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=state.card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=state.skip_step + 1 * (new_state_counter == 3),
    )

    return jax.lax.switch(
        index=new_state_counter - 8 + 8 * (new_state_counter == 3) - 1 * (new_state_counter == 10),
        branches=[
            # Step 8
            check_dealer_draw,
            # Step 10
            check_winner,
            # Step 3
            lambda op: op[0]
        ],
        operand=(new_state, action, consts)
    )

@jax.jit
def check_dealer_draw(state_action_tuple) -> BlackjackState:
    """ Step 8: Check if dealer should take a card """
    state, action, consts = state_action_tuple

    card_sum = calculate_card_value_sum(state.cards_dealer, consts)
    condition = jnp.array([
        jnp.logical_or(card_sum > 17, jnp.logical_and(card_sum >= 17, jnp.any(jnp.isin(state.cards_dealer, jnp.array([25, 26, 51, 52]))))),
        card_sum > 16
    ], dtype=bool)

    new_state_counter = jax.lax.cond(
        pred=jnp.logical_and(jnp.logical_not(state.cards_dealer_counter == 6), jnp.logical_not(condition[consts.PLAYING_RULE])),
        true_fun=lambda: jnp.array(9).astype(jnp.int32),
        false_fun=lambda: jnp.array(10).astype(jnp.int32)
    )

    return jax.lax.cond(
        pred=new_state_counter == 10,
        true_fun=check_winner,
        false_fun=execute_dealer_draw,
        operand=(state, action, consts)
    )

@jax.jit
def execute_dealer_draw(state_action_tuple) -> BlackjackState:
    """ Step 9: Give dealer a card """
    state, action, consts = state_action_tuple

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)
    # Copy cards_dealer and change the value at index cards_dealer_counter to the value of card
    new_cards_dealer = state.cards_dealer.at[state.cards_dealer_counter].set(card)
    # Increment the cards_dealer_counter
    new_cards_dealer_counter = jnp.array(state.cards_dealer_counter + 1).astype(jnp.int32)
    # Change next step to state 8
    new_state_counter = jnp.array(8).astype(jnp.int32)

    new_skip_state = state.step_counter + 41 + 40 * (state.cards_dealer_counter == 2)

    new_state = BlackjackState(
        player_score=state.player_score,
        player_bet=state.player_bet,
        player_action=state.player_action,
        cards_player=state.cards_player,
        cards_player_counter=state.cards_player_counter,
        cards_dealer=new_cards_dealer,
        cards_dealer_counter=new_cards_dealer_counter,
        step_counter=state.step_counter + 1,
        state_counter=new_state_counter,
        cards_permutation=state.cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=state.key,
        subkey=state.subkey,
        last_round=state.last_round,
        skip_step=new_skip_state
    )

    return new_state

@jax.jit
def check_winner(state_action_tuple) -> BlackjackState:
    """ Step 10: Check winner """
    state, action, consts = state_action_tuple

    hand_player = calculate_card_value_sum(state.cards_player, consts)
    hand_dealer = calculate_card_value_sum(state.cards_dealer, consts)

    bet = state.player_bet
    player_score = state.player_score

    def player_wins1x():
        """ Handles the case where the player wins 1x """
        return (player_score + bet, jnp.array(1).astype(jnp.int32))

    def player_looses():
        """ Handles the case where the player looses and the dealer wins """
        return (player_score - bet, jnp.array(2).astype(jnp.int32))

    def handle_draw():
        """ Handles the draw case based on the playing rules """
        # PLAYING_RULE, determines the playing rules, 0 = casino rules, 1 = private rules
        return jax.lax.cond(
            pred=consts.PLAYING_RULE == 0,
            true_fun=lambda score: (score, jnp.array(3).astype(jnp.int32)), # -> CBJR: player score remains the same
            false_fun=lambda _: player_looses(), # -> PBJR: player looses, dealer wins
            operand=player_score
        )

    #-# check win #-#
    # noinspection PyTypeChecker
    new_player_score, new_last_round = jnp.select(
        condlist=[
            hand_player > 21, # player has more than 21
            jnp.logical_and(jnp.logical_and(hand_player == 21, state.cards_player_counter == 2), jnp.logical_and(hand_dealer == 21, state.cards_dealer_counter == 2)), # player and dealer have a Blackjack
            jnp.logical_and(hand_player == 21, state.cards_player_counter == 2), # only player has a Blackjack
            jnp.logical_or(hand_dealer > 21, jnp.logical_and(consts.PLAYING_RULE == 1, state.cards_player_counter >= 6)), # dealer has more than 21 OR [rule PBJR] player has (more than) 6 cards
            hand_player > hand_dealer, # player has more than dealer
            hand_player < hand_dealer, # player has less than dealer
            hand_player == hand_dealer, # player and dealer have the same points
        ],
        choicelist=jnp.array([
            (state.player_score - state.player_bet, jnp.array(5).astype(jnp.int32)), # -> bust: player looses
            handle_draw(),
            (jnp.floor(state.player_score + (state.player_bet * 1.5)).astype(jnp.int32), jnp.array(4).astype(jnp.int32)), # -> player wins (1.5x)
            player_wins1x(),
            player_wins1x(),
            player_looses(),
            handle_draw()
        ])
    )

    # change the next step to state 1
    new_state_counter = jnp.array(1).astype(jnp.int32)
    new_player_action = jnp.array(0).astype(jnp.int32)

    # Condition for mixing the cards
    condition = jnp.logical_or(consts.CARD_SHUFFLING_RULE == 0, jnp.logical_and(consts.CARD_SHUFFLING_RULE == 1, state.card_permutation_counter >= 34))

    # Generates a new permutation of the cards and keys and resets the counter if necessary
    new_cards_permutation, new_card_permutation_counter, new_key, new_subkey = jax.lax.cond(
        pred=condition,
        true_fun=lambda s: (jax.random.permutation(s.subkey, jnp.arange(1, 53)), jnp.array(0).astype(jnp.int32), *jax.random.split(s.key)),
        false_fun=lambda s: (s.cards_permutation, s.card_permutation_counter, s.key, s.subkey),
        operand=state,
    )

    new_skip_state = state.step_counter + 41

    new_state = BlackjackState(
        player_score=new_player_score,
        player_bet=jnp.minimum(state.player_bet, state.player_score),
        player_action=new_player_action,
        cards_player=state.cards_player,
        cards_player_counter=state.cards_player_counter,
        cards_dealer=state.cards_dealer,
        cards_dealer_counter=state.cards_dealer_counter,
        step_counter=state.step_counter + 1,
        state_counter=new_state_counter,
        cards_permutation=new_cards_permutation,
        card_permutation_counter=new_card_permutation_counter,
        key=new_key,
        subkey=new_subkey,
        last_round=new_last_round,
        skip_step =new_skip_state
    )

    return new_state


class JaxBlackjack(JaxEnvironment[BlackjackState, BlackjackObservation, BlackjackInfo, BlackjackConstants]):
    def __init__(self, consts: BlackjackConstants = None, reward_funcs: list[callable] = None):
        super().__init__()
        self.consts = consts or BlackjackConstants()
        self.frame_stack_size = 4
        self.renderer = BlackjackRenderer()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN
        ]
        self.obs_size = 5


    def reset(self, key=jax.random.PRNGKey(int.from_bytes(os.urandom(3), byteorder='big'))) -> Tuple[BlackjackObservation, BlackjackState]:
        # Resets the game state to the initial state.
        key1, subkey1 = jax.random.split(key)
        key2, subkey2 = jax.random.split(key1)
        state = BlackjackState(
            player_score=self.consts.INITIAL_PLAYER_SCORE,
            player_bet=self.consts.INITIAL_PLAYER_BET,
            player_action=jnp.array(0).astype(jnp.int32),
            cards_player=jnp.array([0, 0, 0, 0, 0, 0]).astype(jnp.int32),
            cards_player_counter=jnp.array(0).astype(jnp.int32),  # 0
            cards_dealer=jnp.array([0, 0, 0, 0, 0, 0]).astype(jnp.int32),
            cards_dealer_counter=jnp.array(0).astype(jnp.int32),  # 0
            step_counter=jnp.array(0).astype(jnp.int32),  # 0
            state_counter=jnp.array(1).astype(jnp.int32),  # init state 1
            cards_permutation=jnp.array(jax.random.permutation(subkey1, jnp.arange(1, 53))).astype(jnp.int32),
            card_permutation_counter=jnp.array(0).astype(jnp.int32),  # 0
            key=key2,
            subkey=subkey2,
            last_round=jnp.array(0).astype(jnp.int32),
            skip_step=jnp.array(0).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BlackjackState, action: chex.Array) -> Tuple[BlackjackObservation, BlackjackState, float, bool, BlackjackInfo]:
        new_state = jax.lax.cond(
            pred=jnp.logical_or(self.consts.SKIP_FRAMES == 1, state.step_counter >= state.skip_step),
            true_fun= lambda op: jax.lax.switch(
                op[0].state_counter - 1,
                [select_bet_value, draw_initial_cards, select_next_action, execute_hit, execute_stay, execute_double, check_player_bust, check_dealer_draw, execute_dealer_draw, check_winner],
                (op[0], op[1], self.consts)
            ),
            false_fun= lambda op: BlackjackState(
                player_score=op[0].player_score,
                player_bet=op[0].player_bet,
                player_action=op[0].player_action,
                cards_player=op[0].cards_player,
                cards_player_counter=op[0].cards_player_counter,
                cards_dealer=op[0].cards_dealer,
                cards_dealer_counter=op[0].cards_dealer_counter,
                step_counter=op[0].step_counter + 1,
                state_counter=op[0].state_counter,
                cards_permutation=op[0].cards_permutation,
                card_permutation_counter=op[0].card_permutation_counter,
                key=op[0].key,
                subkey=op[0].subkey,
                last_round=op[0].last_round,
                skip_step=op[0].skip_step
            ),
            operand=(state, action)
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BlackjackState, state: BlackjackState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: BlackjackState, state: BlackjackState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state)
             for reward_func in self.reward_funcs]
        )
        return rewards

    def action_space(self) -> Discrete:
        return Discrete(len(self.action_set))

    def image_space(self) -> Box:
        return Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> Dict:
        return Dict({
            "player_score": Box(0, 1000, (), jnp.int32),
            "player_bet": Box(0, 25, (), jnp.int32),
            "player_current_card_sum": Box(0, 30, (), jnp.int32),
            "dealer_current_card_sum": Box(0, 30, (), jnp.int32),
            "label": Box(0, 8, (), jnp.int32)
        })

    def obs_to_flat_array(self, obs: BlackjackObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_score.flatten(),
            obs.player_bet.flatten(),
            obs.player_current_card_sum.flatten(),
            obs.dealer_current_card_sum.flatten(),
            obs.label.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BlackjackState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BlackjackState):
        player_sum = calculate_card_value_sum(state.cards_player, self.consts)
        # Assuming dealer's first card is the upcard for observation
        dealer_sum = jax.lax.cond(
            pred=state.cards_dealer_counter == 2,
            true_fun=lambda s: calculate_card_value_sum(jnp.concatenate([s.cards_dealer[:1], s.cards_dealer[1:]]), self.consts),
            false_fun=lambda s: calculate_card_value_sum(s.cards_dealer, self.consts),
            operand=state
        )

        player_score = state.player_score.astype(jnp.int32)
        player_bet = state.player_bet.astype(jnp.int32)
        label = jnp.select(
            condlist=[
                jnp.logical_and(state.last_round == 4, jnp.isin(state.state_counter, jnp.array([1, 10], dtype=jnp.int32))),
                jnp.logical_and(state.last_round == 5, jnp.isin(state.state_counter, jnp.array([1, 10], dtype=jnp.int32))),
                jnp.logical_and(state.player_action == 1, jnp.isin(state.state_counter, jnp.array([3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32))),
                jnp.logical_and(state.player_action == 2, jnp.isin(state.state_counter, jnp.array([3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32))),
                jnp.logical_and(state.last_round == 2, jnp.isin(state.state_counter, jnp.array([1, 10], dtype=jnp.int32))),
                jnp.logical_and(state.last_round == 3, jnp.isin(state.state_counter, jnp.array([1, 10], dtype=jnp.int32))),
                jnp.logical_and(state.player_action == 0, jnp.isin(state.state_counter, jnp.array([3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32))),
                jnp.logical_and(state.last_round == 1, jnp.isin(state.state_counter, jnp.array([1, 10], dtype=jnp.int32))),
            ],
            # 0 = no label, 1 = BJ, 2 = Bust, 3 = Double, 4 = Hit, 5 = Lose, 6 = Tie, 7 = Stay, 8 = Win
            choicelist=[1, 2, 3, 4, 5, 6, 7, 8],
            default=0
        ).astype(jnp.int32)

        return BlackjackObservation(
            player_score=player_score,
            player_bet=player_bet,
            player_current_card_sum=player_sum,
            dealer_current_card_sum=dealer_sum,
            label=label
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BlackjackState, all_rewards: jnp.ndarray = None) -> BlackjackInfo:
        return BlackjackInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BlackjackState) -> bool:
        player_has_no_money = state.player_score <= 0
        player_wins = state.player_score >= 1000
        return jnp.logical_or(player_has_no_money, player_wins)


def load_sprites():
    # Load all sprites required for Blackjack rendering
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load background sprite
    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/blackjack/background.npy"), transpose=False)
    SPRITE_BACKGROUND = aj.get_sprite_frame(jnp.expand_dims(background, axis=0), 0)

    # Load empty card sprite
    card_empty = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/blackjack/card_empty.npy"), transpose=False)
    SPRITE_CARD_EMPTY = aj.get_sprite_frame(jnp.expand_dims(card_empty, axis=0), 0)

    # Load questionmark sprite
    questionmark = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/blackjack/questionmark.npy"), transpose=False)
    SPRITE_QUESTIONMARK = aj.get_sprite_frame(jnp.expand_dims(questionmark, axis=0), 0)

    # Load all label sprites
    labels = ["stay", "dble", "hit", "win", "lose", "tie", "bj", "bust"]
    labels_array = []
    for label in labels:
        path = os.path.join(MODULE_DIR, "sprites/blackjack/labels/" + label + ".npy")
        frame = aj.loadFrame(path, transpose=False)
        labels_array.append(aj.get_sprite_frame(jnp.expand_dims(frame, axis=0), 0))
    SPRITES_LABELS = jnp.array(labels_array)

    # Load all black card sprites
    numbers = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "K", "Q", "A"]
    black_cards = []
    for number in numbers:
        path = os.path.join(MODULE_DIR, "sprites/blackjack/cards/black/card_black_" + number + ".npy")
        frame = aj.loadFrame(path, transpose=False)
        black_cards.append(aj.get_sprite_frame(jnp.expand_dims(frame, axis=0), 0))
    SPRITES_BLACK_CARDS = jnp.array(black_cards)

    # Load all red card sprites
    red_cards = []
    for number in numbers:
        path = os.path.join(MODULE_DIR, "sprites/blackjack/cards/red/card_red_" + number + ".npy")
        frame = aj.loadFrame(path)
        red_cards.append(aj.get_sprite_frame(jnp.expand_dims(frame, axis=0), 0))
    SPRITES_RED_CARDS = jnp.array(red_cards)

    # Load digits for player score
    PLAYER_SCORE_DIGIT_SPRITES = aj.load_and_pad_digits(os.path.join(
        MODULE_DIR, "sprites/blackjack/digits/white/digit_white_{}.npy"), num_chars=10)

    # Load digits for sum of card values of the player
    BET_DIGIT_SPRITES = aj.load_and_pad_digits(os.path.join(
        MODULE_DIR, "sprites/blackjack/digits/black/digit_black_{}.npy"), num_chars=10)

    return (
        SPRITE_BACKGROUND,
        SPRITES_LABELS,
        SPRITE_CARD_EMPTY,
        SPRITE_QUESTIONMARK,
        SPRITES_BLACK_CARDS,
        SPRITES_RED_CARDS,
        PLAYER_SCORE_DIGIT_SPRITES,
        BET_DIGIT_SPRITES
    )


class BlackjackRenderer(JAXGameRenderer):
    def __init__(self, consts: BlackjackConstants = None):
        super().__init__()
        self.consts = consts or BlackjackConstants()
        (
            self.SPRITE_BACKGROUND,
            self.SPRITES_LABELS,
            self.SPRITE_CARD_EMPTY,
            self.SPRITE_QUESTIONMARK,
            self.SPRITES_BLACK_CARDS,
            self.SPRITES_RED_CARDS,
            self.PLAYER_SCORE_DIGIT_SPRITES,
            self.BET_DIGIT_SPRITES
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BlackjackState):
        """ Responsible for the graphical representation of the game """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        # Render background - (0, 0) is top-left corner
        raster = aj.render_at(raster, 0, 0, self.SPRITE_BACKGROUND)

        # Render questionmark, when Player can select bet amount
        raster = jax.lax.cond(
            pred = state.state_counter == 1,
            true_fun = lambda r: aj.render_at(r, 37, 59, self.SPRITE_QUESTIONMARK),
            false_fun = lambda r: r,
            operand=raster
        )

        # Render player_score
        # 1. Get digit array (always 4 digits)
        player_score_digits = aj.int_to_digits(state.player_score, max_digits=4)

        # 2. Determine parameters for player score rendering
        player_score_conditions = jnp.array([
            state.player_score < 10,
            jnp.logical_and(state.player_score >= 10, state.player_score < 100),
            jnp.logical_and(state.player_score >= 100, state.player_score < 1000),
            state.player_score >= 1000
        ], dtype=bool)
        # Start at index 3 if single, 2 if double, 1 if triple, 0 if quadrupel
        player_score_start_index = jnp.select(player_score_conditions, jnp.array([3, 2, 1, 0]))
        # Render 1 digit if single, 2 if double, 3 if triple, 4 if quadrupel
        player_score_num_to_render = jnp.select(player_score_conditions, jnp.array([1, 2, 3, 4]))

        # 3. Render player score using the selective renderer
        raster = aj.render_label_selective(raster, 41, 44,
                                           player_score_digits, self.PLAYER_SCORE_DIGIT_SPRITES,
                                           player_score_start_index, player_score_num_to_render,
                                           spacing=5)


        # Render player_bet
        # 1. Get digit array (always 2 digits)
        player_bet_digits = aj.int_to_digits(state.player_bet, max_digits=2)

        # 2. Determine parameters for player score rendering
        is_player_bet_single_digit = state.player_score < 10
        player_bet_start_index = jax.lax.select(is_player_bet_single_digit, 1, 0)  # Start at index 1 if single, 0 if double
        player_bet_num_to_render = jax.lax.select(is_player_bet_single_digit, 1, 2)  # Render 1 digit if single, 2 if double

        # 3. Render player bet using the selective renderer
        raster = aj.render_label_selective(raster, 45, 59,
                                           player_bet_digits, self.BET_DIGIT_SPRITES,
                                           player_bet_start_index, player_bet_num_to_render,
                                           spacing=5)


        # Render player action in step 3, 4, 5, 6, 7, 8, 9
        raster = jnp.select(
            condlist=[
                jnp.logical_and(state.state_counter >= 3, state.state_counter <= 9),
                jnp.logical_and(state.state_counter != 2, state.last_round != 0)
            ],
            choicelist=[
                aj.render_at(raster, 36, 74, self.SPRITES_LABELS[state.player_action]),
                aj.render_at(raster, 36, 74, self.SPRITES_LABELS[3 + state.last_round - 1])
            ],
            default=raster
        )

        def get_card_sprite(card_number):
            """ Calculates the sprite to the given number """
            sprite = jax.lax.cond(
                pred = jnp.less_equal(card_number / 13, 2),
                true_fun = lambda cn: self.SPRITES_BLACK_CARDS[(cn - 1) % 13],
                false_fun = lambda cn: self.SPRITES_RED_CARDS[(cn - 1) % 13],
                operand=card_number
            )

            return sprite


        # Render cards of player
        raster = jax.lax.fori_loop(
            lower = 0,
            upper = state.cards_player_counter,
            body_fun = lambda i, val: aj.render_at(val, 44, 88 + i * 19, get_card_sprite(state.cards_player[i])),
            init_val = raster
        )

        # Render cards of the dealer
        raster = jnp.select(
            condlist=[
                jnp.logical_and(state.cards_dealer_counter == 3, state.skip_step - state.step_counter >= 40),
                jnp.logical_and(state.cards_dealer_counter == 2, jnp.logical_not(jnp.isin(state.state_counter, jnp.array([8, 9, 10, 1], dtype=jnp.int32))))
            ],
            choicelist=[
                aj.render_at(aj.render_at(raster, 44, 3, get_card_sprite(state.cards_dealer[0])), 76, 3, get_card_sprite(state.cards_dealer[1]), 0),
                aj.render_at(aj.render_at(raster, 44, 3, get_card_sprite(state.cards_dealer[0])), 76, 3, self.SPRITE_CARD_EMPTY, 0),
            ],
            default=jax.lax.fori_loop(
                    lower=0,
                    upper=state.cards_dealer_counter,
                    body_fun=lambda i, val: jax.lax.cond(
                        pred=i <= 2,
                        true_fun=lambda: aj.render_at(val, 44 + i * 32, 3, get_card_sprite(state.cards_dealer[i])),
                        false_fun=lambda: aj.render_at(val, 44 + (i - 3) * 32, 22, get_card_sprite(state.cards_dealer[i]))
                    ),
                    init_val=raster
                ),
        )

        return raster