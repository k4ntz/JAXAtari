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
import pygame
import argparse

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.spaces import Space, Discrete, Box, Dict


class BlackjackConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    # Determines when the cards are being shuffled, 0 = after every round, 1 = after drawing 34 cards
    CARD_SHUFFLING_RULE = 0
    # Determines the playing rules, 0 = Casino Rules, 1 = Private Rules
    PLAYING_RULE = 0
    INITIAL_PLAYER_SCORE = jnp.array(200).astype(jnp.int32)
    INITIAL_PLAYER_BET = jnp.array(1).astype(jnp.int32)


class BlackjackState(NamedTuple):
    # The score of the player
    player_score: chex.Array
    # The bet of the player in this round (between 1 and 25)
    player_bet: chex.Array

    # Tells whether hit stay or double is selected 0 = stay, 1 = double, 2 = hit
    player_action: chex.Array
    # An array which includes all the cards of the player
    # Format: [n, n, 0, 0, 0, 0] (Array has a size of 6)
    # 0 -> no card currently
    # n Values: 01 -> B2
    #           02 -> B2
    #           03 -> B3
    #           04 -> B3
    #           ...
    #           17 -> B10
    #           18 -> B10
    #           19 -> BJ
    #           20 -> BJ
    #           21 -> BK
    #           22 -> BK
    #           23 -> BQ
    #           24 -> BQ
    #           25 -> BA
    #           26 -> BA
    #           27 -> R2
    #           28 -> R2
    #           29 -> R3
    #           30 -> R3
    #           ...
    #           43 -> R10
    #           44 -> R10
    #           45 -> RJ
    #           46 -> RJ
    #           47 -> RK
    #           48 -> RK
    #           49 -> RQ
    #           50 -> RQ
    #           51 -> RA
    #           52 -> RA
    cards_player: chex.Array
    # The counter for the index which is free
    cards_player_counter: chex.Numeric
    # An array which includes all the of the dealer (same format as player)
    cards_dealer: chex.Array
    # The counter for the index which is free
    cards_dealer_counter: chex.Numeric
    # The counter for the current step
    step_counter: chex.Numeric
    # A counter used for counting the state in the step Method
    state_counter: chex.Numeric
    # An array which contains the permutation of the cards which can be drawn
    # The array has a size of 26
    # Format look at cards_player
    cards_permutation: chex.Array
    # A counter which tells at which position of the index the next number can be taken
    card_permutation_counter: chex.Numeric
    # The seed for generating random numbers
    seed: chex.Numeric
    # indicates whether the last round was won (1) lost(2), draw(3), blackjack(4), bust(5) or there was no last round (0)
    last_round: chex.Array


class BlackjackObservation(NamedTuple):
    player_score: chex.Array
    player_bet: chex.Array
    player_current_card_sum: chex.Array
    dealer_current_card_sum: chex.Array


class BlackjackInfo(NamedTuple):
    player_score: chex.Array
    step_counter: chex.Array


@jax.jit
def calculate_card_value_sum(hand: chex.Array) -> int:
    """ Calculates the card value of all cards in hand """
    def sum_cards(i, param):
        """ Adds the current card value accept aces to the sum and updates the number of aces """

        csum = param[0]
        number_of_A = param[1]
        value = hand[i]

        # Checks the value for the current card and adds it to csum
        # Aces are ignored
        card_conditions = jnp.array([
            # 1, 2, 27, 28 is a card with value 2
            jnp.logical_or(jnp.logical_or(value == 1, value == 2), jnp.logical_or(value == 27, value == 28)),
            # 3, 4, 29, 30 is a card with value 3
            jnp.logical_or(jnp.logical_or(value == 3, value == 4), jnp.logical_or(value == 29, value == 30)),
            # 5, 6, 31, 32 is a card with value 4
            jnp.logical_or(jnp.logical_or(value == 5, value == 6), jnp.logical_or(value == 31, value == 32)),
            # 7, 8, 33, 34 is a card with value 5
            jnp.logical_or(jnp.logical_or(value == 7, value == 8), jnp.logical_or(value == 33, value == 34)),
            # 9, 10, 35, 36 is a card with value 6
            jnp.logical_or(jnp.logical_or(value == 9, value == 10), jnp.logical_or(value == 35, value == 36)),
            # 11, 12, 37, 38 is a card with value 7
            jnp.logical_or(jnp.logical_or(value == 11, value == 12), jnp.logical_or(value == 37, value == 38)),
            # 13, 14, 39, 40 is a card with value 8
            jnp.logical_or(jnp.logical_or(value == 13, value == 14), jnp.logical_or(value == 39, value == 40)),
            # 15, 16, 41, 42 is a card with value 9
            jnp.logical_or(jnp.logical_or(value == 15, value == 16), jnp.logical_or(value == 41, value == 42)),
            # 17, 18, 19, 20, 21, 22, 23, 24, 43, 44, 45, 46, 47, 48, 49, 50 is a card with value 10
            jnp.logical_or(
                jnp.logical_or(
                    jnp.logical_or(jnp.logical_or(value == 17, value == 18), jnp.logical_or(value == 19, value == 20)),
                    jnp.logical_or(jnp.logical_or(value == 21, value == 22), jnp.logical_or(value == 23, value == 24))
                ),
                jnp.logical_or(
                    jnp.logical_or(jnp.logical_or(value == 43, value == 44), jnp.logical_or(value == 45, value == 46)),
                    jnp.logical_or(jnp.logical_or(value == 47, value == 48), jnp.logical_or(value == 49, value == 50)),
                ),
            ),
            # 25, 26, 51, 52 is an A card
            jnp.logical_or(jnp.logical_or(value == 25, value == 26), jnp.logical_or(value == 51, value == 52))
        ], dtype=bool)
        card_values = jnp.array([
            csum + 2,
            csum + 3,
            csum + 4,
            csum + 5,
            csum + 6,
            csum + 7,
            csum + 8,
            csum + 9,
            csum + 10,
            csum
        ])
        csum = jnp.select(card_conditions, card_values, default = csum)

        # Checks if the current card is an ace
        number_of_A = jnp.where(jnp.logical_or(jnp.logical_or(value == 25, value == 26), jnp.logical_or(value == 51, value == 52)), number_of_A + 1, number_of_A)

        return csum, number_of_A

    # Loops over all cards and returns the sum of their values without the aces and the number of aces
    card_sum, number_of_aces = jax.lax.fori_loop(
        init_val=(0, 0),
        upper=len(hand),
        lower=0,
        body_fun=sum_cards
    )

    def calculate_value_for_A(csum, number_of_A):
        """ Decides which value an Ace gets """
        number_of_A = number_of_A - 1
        csum = csum + number_of_A
        csum = jax.lax.cond(
            pred=csum > 10,
            true_fun=lambda: csum + 1,
            false_fun=lambda: csum + 11,
        )
        return csum

    # Calculates the value for the Aces if there is more than one Ace
    card_sum = jax.lax.cond(
        pred=number_of_aces > 0,
        true_fun=lambda: calculate_value_for_A(card_sum, number_of_aces),
        false_fun=lambda: card_sum
    )

    return card_sum

@jax.jit
def step_1(state_action_tuple):
    """ Step 1: Select bet value with Up/Down and confirm with Fire (and reset displayed cards) """
    state, action, consts = state_action_tuple

    # Check if the action is Up, Down or Fire
    up = action == Action.UP
    down = action == Action.DOWN
    fire = action == Action.FIRE

    # If Up is pressed, increase bet by 1
    new_player_bet = jax.lax.cond(
        pred=jnp.logical_and(up, state.player_bet < state.player_score),
        true_fun=lambda: jnp.minimum(state.player_bet + 1, 25),
        false_fun=lambda: state.player_bet
    )

    # If Down is pressed, decrease bet by 1
    new_player_bet = jax.lax.cond(
        pred=down,
        true_fun= lambda: jnp.maximum(state.player_bet - 1, 1),
        false_fun= lambda: new_player_bet
    )

    # Set the counter for the state to 2 if Fire is pressed, otherwise keep it as is
    new_state_counter, new_cards_player, new_cards_player_counter, new_cards_dealer, new_cards_dealer_counter = jax.lax.cond(
        pred=fire,
        true_fun=lambda: (2, jnp.array([0, 0, 0, 0, 0, 0]).astype(jnp.int32), jnp.array(0).astype(jnp.int32), jnp.array([0, 0, 0, 0, 0, 0]).astype(jnp.int32), jnp.array(0).astype(jnp.int32)),
        false_fun=lambda: (state.state_counter, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter)
    )

    return state.player_score, new_player_bet, state.player_action, new_cards_player, new_cards_player_counter, new_cards_dealer, new_cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, state.card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_2(state_action_tuple):
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
    new_cards_player = state.cards_player.at[state.cards_player_counter].set(card_player1)
    new_cards_player = new_cards_player.at[state.cards_player_counter+1].set(card_player2)
    # Increment the cards_player_counter
    new_cards_player_counter = jnp.array(
        state.cards_player_counter + 2).astype(jnp.int32)

    # Give out the cards to the dealer
    new_cards_dealer = state.cards_dealer.at[state.cards_dealer_counter].set(card_dealer1)
    new_cards_dealer = new_cards_dealer.at[state.cards_dealer_counter+1].set(card_dealer2)
    new_cards_dealer_counter = jnp.array(
        state.cards_dealer_counter + 2).astype(jnp.int32)


    # Set state to 3 or to the final state if the dealer draws a blackjack
    new_state_counter = jax.lax.cond(
        pred = jnp.logical_or(calculate_card_value_sum(new_cards_dealer) == 21, calculate_card_value_sum(new_cards_player) == 21),
        true_fun = lambda: jnp.array(10).astype(jnp.int32),
        false_fun = lambda: jnp.array(3).astype(jnp.int32))


    # Returns all modified and unmodified attributes of the state
    return state.player_score, state.player_bet, state.player_action, new_cards_player, new_cards_player_counter, new_cards_dealer, new_cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, new_card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_3(state_action_tuple):
    """ Step 3: Select hit, stay or double with Up/Down and confirm with Fire """
    state, action, consts = state_action_tuple

    new_player_action = jnp.array(jax.lax.cond(
        pred=action == Action.UP,
        true_fun=lambda: jax.lax.cond(
            pred=state.player_action != 2,
            true_fun=lambda:
                jax.lax.cond(
                pred=jnp.logical_and(state.player_action == 0, jnp.logical_or(jnp.logical_or((state.player_bet * 2 > state.player_score), jnp.logical_or(jnp.logical_and(jnp.logical_and(calculate_card_value_sum(state.cards_player) != 10, calculate_card_value_sum(state.cards_player) != 11), consts.PLAYING_RULE == 0), consts.PLAYING_RULE == 1)), state.cards_player_counter > 2 )),
                true_fun=lambda: state.player_action + 2,
                false_fun=lambda: state.player_action + 1,
            ),
            false_fun=lambda: state.player_action,
            ),
        false_fun=lambda: jax.lax.cond(
                pred=action == Action.DOWN,
                true_fun=lambda:
                    jax.lax.cond(
                        pred=state.player_action != 0,
                        true_fun=lambda:
                            jax.lax.cond(
                                pred=jnp.logical_and(state.player_action == 2, jnp.logical_or(jnp.logical_or((state.player_bet * 2 > state.player_score), jnp.logical_or(jnp.logical_and(jnp.logical_and(calculate_card_value_sum(state.cards_player) != 10, calculate_card_value_sum(state.cards_player) != 11), consts.PLAYING_RULE == 0), consts.PLAYING_RULE == 1)), state.cards_player_counter > 2 )),
                                true_fun=lambda: state.player_action - 2,
                                false_fun=lambda: state.player_action - 1,
                        ),
                        false_fun=lambda: state.player_action,
                ),
                false_fun=lambda: state.player_action,
            ),
        )).astype(jnp.int32)

    new_state_counter = jnp.array(jax.lax.cond(
        pred=action == Action.FIRE,
        true_fun=lambda: jax.lax.cond(
            pred=state.player_action == 0,
            true_fun=lambda: 5,
            false_fun=lambda: jax.lax.cond(
                pred=state.player_action == 1,
                true_fun=lambda: 6,
                false_fun=lambda: 4,
            ),
        ),
        false_fun=lambda: state.state_counter,
    )).astype(jnp.int32)
    new_state_counter = jnp.array(jax.lax.cond(
        pred=state.cards_player_counter >= 6,
        true_fun=lambda: 10,
        false_fun=lambda: new_state_counter,
    )).astype(jnp.int32)

    return state.player_score, state.player_bet, new_player_action, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, state.card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_4(state_action_tuple):
    """ Step 4: Execute hit """
    state, action, consts = state_action_tuple

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)
    # Copy cards_player and change the value at index cards_player_counter to the value of card
    new_cards_player = state.cards_player.at[state.cards_player_counter].set(card)
    # Increment the cards_player_counter
    new_cards_player_counter = jnp.array(
        state.cards_player_counter + 1).astype(jnp.int32)
    # Change next step to state 7
    new_state_counter = jnp.array(7).astype(jnp.int32)
    # Returns all modified and unmodified attributes of the state
    return state.player_score, state.player_bet, state.player_action, new_cards_player, new_cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, new_card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_5(state_action_tuple):
    """ Step 5: Execute stay """
    state, action, consts = state_action_tuple

    # change the next step to state 8
    new_state_counter = jnp.array(8).astype(jnp.int32)
    return state.player_score, state.player_bet, state.player_action, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, state.card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_6(state_action_tuple):
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
    new_cards_player_counter = jnp.array(
        state.cards_player_counter + 1).astype(jnp.int32)

    # check hand, change the next step
    new_state_counter = jax.lax.cond(
        pred=calculate_card_value_sum(new_cards_player) > 21,
        true_fun=lambda: jnp.array(10).astype(jnp.int32),  # > 21 -> lose
        # <= 21 -> dealer checks his cards
        false_fun=lambda: jnp.array(8).astype(jnp.int32)
    )

    return state.player_score, new_player_bet, state.player_action, new_cards_player, new_cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, new_card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_7(state_action_tuple):
    """ Step 7: Check if player has 21 or more """
    state, action, consts = state_action_tuple

    new_state = jnp.array(jax.lax.cond(
        pred=calculate_card_value_sum(state.cards_player) >= 21,
        true_fun=lambda: jax.lax.cond(
            pred=calculate_card_value_sum(state.cards_player) == 21,
            true_fun=lambda:8,
            false_fun=lambda: 10,
        ),
        false_fun=lambda: 3,

    )).astype(jnp.int32)
    return state.player_score, state.player_bet, state.player_action, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state, state.cards_permutation, state.card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_8(state_action_tuple):
    """" Step 8: Check if dealer should take a card """
    state, action, consts = state_action_tuple

    card_sum = calculate_card_value_sum(state.cards_dealer)

    condition = jnp.array([
        jnp.logical_or(card_sum > 17, jnp.logical_and(card_sum >= 17, jnp.any(jnp.isin(state.cards_dealer, jnp.array([25, 26, 51, 52]))))),
        card_sum > 16
    ], dtype=bool)

    new_state_counter = jax.lax.cond(
        pred=state.cards_dealer_counter == 6,
        true_fun=lambda: jnp.array(10).astype(jnp.int32),
        false_fun=lambda: jax.lax.cond(
            pred=condition[consts.PLAYING_RULE],
            true_fun=lambda: jnp.array(10).astype(jnp.int32),
            false_fun=lambda: jnp.array(9).astype(jnp.int32)
        ),
    )
    return state.player_score, state.player_bet, state.player_action, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, state.card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_9(state_action_tuple):
    """ Step 9: Give dealer a card """
    state, action, consts = state_action_tuple

    # Sample a new card from the permutation array
    card = state.cards_permutation[state.card_permutation_counter]
    # Increment counter of permutation array
    new_card_permutation_counter = jnp.array(state.card_permutation_counter + 1).astype(jnp.int32)
    # Copy cards_dealer and change the value at index cards_dealer_counter to the value of card
    new_cards_dealer = state.cards_dealer.at[state.cards_dealer_counter].set(card)
    # Increment the cards_dealer_counter
    new_cards_dealer_counter = jnp.array(
        state.cards_dealer_counter + 1).astype(jnp.int32)
    # Change next step to state 8
    new_state_counter = jnp.array(8).astype(jnp.int32)
    # Returns all modified and unmodified attributes of the state
    return state.player_score, state.player_bet, state.player_action, state.cards_player, state.cards_player_counter, new_cards_dealer, new_cards_dealer_counter, state.step_counter, new_state_counter, state.cards_permutation, new_card_permutation_counter, state.seed, state.last_round


@jax.jit
def step_10(state_action_tuple):
    """ Step 10: Check winner """
    state, action, consts = state_action_tuple

    hand_player = calculate_card_value_sum(state.cards_player)
    hand_dealer = calculate_card_value_sum(state.cards_dealer)

    bet = state.player_bet
    player_score_int = state.player_score

    # IF player hat > 21 ODER versucht MEHR ALS 6 karten zu ziehen
    # TRUE:     -> player looses (0x)
    # FALSE:    IF player hat 21
    #           TRUE:   IF dealer hat 21
    #                   TRUE:   -> draw (see ruleset)
    #                   FALSE:  -> player wins (1.5x)
    #           FALSE:  IF dealer has > 21
    #                   TRUE:   -> player wins (1x)
    #                   FALSE:  IF player has > dealer
    #                           TRUE:   -> player wins (1x)
    #                           FALSE:  IF player has < dealer
    #                                   TRUE:   -> player looses (0x)
    #                                   FALSE:  -> draw (see ruleset)

    def player_wins1x():
        """ Handles the case where the player wins 1x """
        return (player_score_int + bet, jnp.array(1).astype(jnp.int32))

    def player_looses():
        """ Handles the case where the player looses and the dealer wins """
        return (player_score_int - bet, jnp.array(2).astype(jnp.int32))

    def handle_draw():
        """ Handles the draw case based on the playing rules """
        # PLAYING_RULE, determines the playing rules, 0 = casino rules, 1 = private rules
        return jax.lax.cond(
            pred=consts.PLAYING_RULE == 0,
            true_fun=lambda: (player_score_int, jnp.array(3).astype(jnp.int32)), # -> CBJR: player score remains the same
            false_fun=player_looses, # -> PBJR: player looses, dealer wins
        )

    #-# check win #-#
    # new_last_round: indicates whether the last round was won (1) lost(2), draw(3), blackjack(4), bust(5) or there was no last round (0)
    new_player_score_int, new_last_round = jax.lax.cond(
        # player has more than 21
        pred=hand_player > 21,
        true_fun=lambda: (player_score_int - bet, jnp.array(5).astype(jnp.int32)), # -> bust: player looses, dealer wins
        false_fun=lambda: jax.lax.cond(
            # player has blackjack
            pred=jnp.logical_and(hand_player == 21, state.cards_player_counter == 2),
            true_fun=lambda: jax.lax.cond(
                # dealer has also blackjack
                pred=jnp.logical_and(hand_dealer == 21, state.cards_dealer_counter == 2),
                true_fun=handle_draw, # -> draw (see ruleset)
                false_fun=lambda: (jnp.floor(player_score_int + (bet * 1.5)).astype(jnp.int32), jnp.array(4).astype(jnp.int32) ), # -> player wins (1.5x)
            ),
            false_fun=lambda: jax.lax.cond(
                # dealer has more than 21 OR [rule PBJR] player has (more than) 6 cards
                pred=jnp.logical_or(hand_dealer > 21, 
                                    jnp.logical_and(consts.PLAYING_RULE == 1, state.cards_player_counter >= 6)
                     ),
                true_fun=player_wins1x, # -> player wins (1x)
                false_fun=lambda: jax.lax.cond(
                    # player has more than dealer
                    pred=hand_player > hand_dealer,
                    true_fun=player_wins1x, # -> player wins (1x)
                    false_fun=lambda: jax.lax.cond(
                        # player has less than dealer
                        pred=hand_player < hand_dealer,
                        true_fun=player_looses, # -> player looses, dealer wins
                        false_fun=handle_draw, # -> draw (see ruleset)
                    ),
                ),
            ),
        )
    )

    # change player score (money stash)
    new_player_score = jnp.array(new_player_score_int).astype(jnp.int32)

    # change the next step to state 1
    new_state_counter = jnp.array(1).astype(jnp.int32)
    new_player_action = jnp.array(0).astype(jnp.int32)

    # Generates a new seed if necessary
    new_seed = jax.lax.cond(
        pred=jnp.logical_or(consts.CARD_SHUFFLING_RULE == 0, jnp.logical_and(consts.CARD_SHUFFLING_RULE == 1, state.card_permutation_counter >= 34)),
        true_fun=lambda: jnp.array(state.seed + 1).astype(jnp.int32),
        false_fun=lambda: state.seed
    )

    # Generates a new permutation of the cards and resets the counter if necessary
    new_cards_permutation, new_card_permutation_counter = jax.lax.cond(
        pred=jnp.logical_or(consts.CARD_SHUFFLING_RULE == 0, jnp.logical_and(consts.CARD_SHUFFLING_RULE == 1, state.card_permutation_counter >= 34)),
        true_fun=lambda: (jnp.array(jax.random.permutation(jax.random.key(new_seed), jnp.arange(1, 53))).astype(jnp.int32),
                          jnp.array(0).astype(jnp.int32)),
        false_fun=lambda: (state.cards_permutation, state.card_permutation_counter)
    )

    return new_player_score, jnp.minimum(state.player_bet, jnp.minimum(25, new_player_score_int)), new_player_action, state.cards_player, state.cards_player_counter, state.cards_dealer, state.cards_dealer_counter, state.step_counter, new_state_counter, new_cards_permutation, new_card_permutation_counter, new_seed, new_last_round


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
        self.obs_size = 4

    def reset(self, key=None) -> Tuple[BlackjackObservation, BlackjackState]:
        # Resets the game state to the initial state.
        seed = int.from_bytes(os.urandom(3), byteorder='big')
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
            cards_permutation=jnp.array(jax.random.permutation(
                jax.random.key(seed), jnp.arange(1, 53))).astype(jnp.int32),
            card_permutation_counter=jnp.array(0).astype(jnp.int32),  # 0
            seed=seed,
            last_round=jnp.array(0).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BlackjackState, action: chex.Array) -> Tuple[BlackjackObservation, BlackjackState, float, bool, BlackjackInfo]:
        new_player_score, new_player_bet, new_player_action, new_cards_player, new_cards_player_counter, new_cards_dealer, new_cards_dealer_counter, new_step_counter, new_state_counter, new_card_permutation, new_card_permutation_counter, new_seed, new_last_round = jax.lax.switch(
            state.state_counter - 1,
            [step_1, step_2, step_3, step_4, step_5, step_6, step_7, step_8, step_9, step_10],
            (state, action, self.consts)
        )

        new_state = BlackjackState(
            player_score=new_player_score,
            player_bet=new_player_bet,
            player_action=new_player_action,
            cards_player=new_cards_player,
            cards_player_counter=new_cards_player_counter,
            cards_dealer=new_cards_dealer,
            cards_dealer_counter=new_cards_dealer_counter,
            step_counter=new_step_counter,
            state_counter=new_state_counter,
            cards_permutation=new_card_permutation,
            card_permutation_counter=new_card_permutation_counter,
            seed=new_seed,
            last_round=new_last_round

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

    def action_space(self) -> Space:
        return Discrete(4)

    def image_space(self) -> Space:
        return Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> Space:
        return Dict({
            "player_score": Box(0, 1000, shape=(1,)),
            "player_bet": Box(0, 25, shape=(1,)),
            "player_current_card_sum": Box(0, 30, shape=(1,)),
            "dealer_current_card_sum": Box(0, 30, shape=(1,))
        })

    def obs_to_flat_array(self, obs: BlackjackObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_score.flatten(),
            obs.player_bet.flatten(),
            obs.player_current_card_sum.flatten(),
            obs.dealer_current_card_sum.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BlackjackState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BlackjackState):
        player_sum = calculate_card_value_sum(state.cards_player)
        # Assuming dealer's first card is the upcard for observation
        dealer_sum = jax.lax.cond(
            pred=state.cards_dealer_counter == 2,
            true_fun=lambda: calculate_card_value_sum(jnp.concatenate([state.cards_dealer[:1], state.cards_dealer[1:]])),
            false_fun=lambda: calculate_card_value_sum(state.cards_dealer)
        )

        return BlackjackObservation(
            player_score=state.player_score,
            player_bet=state.player_bet,
            player_current_card_sum=player_sum,
            dealer_current_card_sum=dealer_sum
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BlackjackState, all_rewards: jnp.ndarray) -> BlackjackInfo:
        return BlackjackInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BlackjackState) -> bool:
        player_has_no_money = state.player_score <= 0
        player_wins = state.player_score >= 1000
        return jnp.logical_or(player_has_no_money, player_wins)


def load_sprites():
    # Load all sprites required for Blackjack rendering
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    background = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/background.npy"), transpose=False)

    bj = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/bj.npy"), transpose=False)
    bust = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/bust.npy"), transpose=False)
    dble = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/dble.npy"), transpose=False)
    hit = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/hit.npy"), transpose=False)
    lose = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/lose.npy"), transpose=False)
    stay = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/stay.npy"), transpose=False)
    win = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/win.npy"), transpose=False)
    tie = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/tie.npy"), transpose=False)

    questionmark = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/questionmark.npy"), transpose=False)

    card_empty = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_empty.npy"), transpose=False)

    card_black_2 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_2.npy"), transpose=False)
    card_black_3 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_3.npy"), transpose=False)
    card_black_4 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_4.npy"), transpose=False)
    card_black_5 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_5.npy"), transpose=False)
    card_black_6 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_6.npy"), transpose=False)
    card_black_7 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_7.npy"), transpose=False)
    card_black_8 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_8.npy"), transpose=False)
    card_black_9 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_9.npy"), transpose=False)
    card_black_10 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_10.npy"), transpose=False)
    card_black_J = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_J.npy"), transpose=False)
    card_black_K = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_K.npy"), transpose=False)
    card_black_Q = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_Q.npy"), transpose=False)
    card_black_A = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_black_A.npy"), transpose=False)

    card_red_2 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_2.npy"), transpose=False)
    card_red_3 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_3.npy"), transpose=False)
    card_red_4 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_4.npy"), transpose=False)
    card_red_5 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_5.npy"), transpose=False)
    card_red_6 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_6.npy"), transpose=False)
    card_red_7 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_7.npy"), transpose=False)
    card_red_8 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_8.npy"), transpose=False)
    card_red_9 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_9.npy"), transpose=False)
    card_red_10 = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_10.npy"), transpose=False)
    card_red_J = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_J.npy"), transpose=False)
    card_red_K = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_K.npy"), transpose=False)
    card_red_Q = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_Q.npy"), transpose=False)
    card_red_A = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/blackjack/card_red_A.npy"), transpose=False)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BACKGROUND = jnp.expand_dims(background, axis=0)

    SPRITE_BJ = jnp.expand_dims(bj, axis=0)
    SPRITE_BUST = jnp.expand_dims(bust, axis=0)
    SPRITE_DBLE = jnp.expand_dims(dble, axis=0)
    SPRITE_HIT = jnp.expand_dims(hit, axis=0)
    SPRITE_LOSE = jnp.expand_dims(lose, axis=0)
    SPRITE_STAY = jnp.expand_dims(stay, axis=0)
    SPRITE_WIN = jnp.expand_dims(win, axis=0)
    SPRITE_TIE = jnp.expand_dims(tie, axis=0)

    SPRITE_CARD_EMPTY = jnp.expand_dims(card_empty, axis=0)

    SPRITE_QUESTIONMARK = jnp.expand_dims(questionmark, axis=0)

    SPRITE_CARD_BLACK_2 = jnp.expand_dims(card_black_2, axis=0)
    SPRITE_CARD_BLACK_3 = jnp.expand_dims(card_black_3, axis=0)
    SPRITE_CARD_BLACK_4 = jnp.expand_dims(card_black_4, axis=0)
    SPRITE_CARD_BLACK_5 = jnp.expand_dims(card_black_5, axis=0)
    SPRITE_CARD_BLACK_6 = jnp.expand_dims(card_black_6, axis=0)
    SPRITE_CARD_BLACK_7 = jnp.expand_dims(card_black_7, axis=0)
    SPRITE_CARD_BLACK_8 = jnp.expand_dims(card_black_8, axis=0)
    SPRITE_CARD_BLACK_9 = jnp.expand_dims(card_black_9, axis=0)
    SPRITE_CARD_BLACK_10 = jnp.expand_dims(card_black_10, axis=0)
    SPRITE_CARD_BLACK_J = jnp.expand_dims(card_black_J, axis=0)
    SPRITE_CARD_BLACK_K = jnp.expand_dims(card_black_K, axis=0)
    SPRITE_CARD_BLACK_Q = jnp.expand_dims(card_black_Q, axis=0)
    SPRITE_CARD_BLACK_A = jnp.expand_dims(card_black_A, axis=0)

    SPRITE_CARD_RED_2 = jnp.expand_dims(card_red_2, axis=0)
    SPRITE_CARD_RED_3 = jnp.expand_dims(card_red_3, axis=0)
    SPRITE_CARD_RED_4 = jnp.expand_dims(card_red_4, axis=0)
    SPRITE_CARD_RED_5 = jnp.expand_dims(card_red_5, axis=0)
    SPRITE_CARD_RED_6 = jnp.expand_dims(card_red_6, axis=0)
    SPRITE_CARD_RED_7 = jnp.expand_dims(card_red_7, axis=0)
    SPRITE_CARD_RED_8 = jnp.expand_dims(card_red_8, axis=0)
    SPRITE_CARD_RED_9 = jnp.expand_dims(card_red_9, axis=0)
    SPRITE_CARD_RED_10 = jnp.expand_dims(card_red_10, axis=0)
    SPRITE_CARD_RED_J = jnp.expand_dims(card_red_J, axis=0)
    SPRITE_CARD_RED_K = jnp.expand_dims(card_red_K, axis=0)
    SPRITE_CARD_RED_Q = jnp.expand_dims(card_red_Q, axis=0)
    SPRITE_CARD_RED_A = jnp.expand_dims(card_red_A, axis=0)

    # Load digits for player score
    PLAYER_SCORE_DIGIT_SPRITES = aj.load_and_pad_digits(os.path.join(
        MODULE_DIR, "sprites/blackjack/digit_white_{}.npy"), num_chars=10)

    # Load digits for sum of card values of the player
    BET_DIGIT_SPRITES = aj.load_and_pad_digits(os.path.join(
        MODULE_DIR, "sprites/blackjack/digit_black_{}.npy"), num_chars=10)

    return (
        SPRITE_BACKGROUND,
        SPRITE_BJ,
        SPRITE_BUST,
        SPRITE_DBLE,
        SPRITE_HIT,
        SPRITE_LOSE,
        SPRITE_STAY,
        SPRITE_WIN,
        SPRITE_TIE,
        SPRITE_CARD_EMPTY,
        SPRITE_QUESTIONMARK,
        SPRITE_CARD_BLACK_2,
        SPRITE_CARD_BLACK_3,
        SPRITE_CARD_BLACK_4,
        SPRITE_CARD_BLACK_5,
        SPRITE_CARD_BLACK_6,
        SPRITE_CARD_BLACK_7,
        SPRITE_CARD_BLACK_8,
        SPRITE_CARD_BLACK_9,
        SPRITE_CARD_BLACK_10,
        SPRITE_CARD_BLACK_J,
        SPRITE_CARD_BLACK_K,
        SPRITE_CARD_BLACK_Q,
        SPRITE_CARD_BLACK_A,
        SPRITE_CARD_RED_2,
        SPRITE_CARD_RED_3,
        SPRITE_CARD_RED_4,
        SPRITE_CARD_RED_5,
        SPRITE_CARD_RED_6,
        SPRITE_CARD_RED_7,
        SPRITE_CARD_RED_8,
        SPRITE_CARD_RED_9,
        SPRITE_CARD_RED_10,
        SPRITE_CARD_RED_J,
        SPRITE_CARD_RED_K,
        SPRITE_CARD_RED_Q,
        SPRITE_CARD_RED_A,
        PLAYER_SCORE_DIGIT_SPRITES,
        BET_DIGIT_SPRITES
    )


def main():
    argparser = argparse.ArgumentParser(description="Difficulty Settings of the Game")
    argparser.add_argument(
        '--difficulty',
        type=int,
        default=0,
        help='Rules of the Game 0=Casino Rules (default), 1=Private Rules'
    )
    argparser.add_argument(
        '--shuffle',
        type=int,
        default=0,
        help='Shuffle Behavior 0=at the end of every round (default), 1=after drawing 34 cards'
    )
    args = argparser.parse_args()
    diff = args.difficulty
    shuff = args.shuffle
    if not (0 <= diff <= 1):
        argparser.error("Difficulty must be between 0 and 1")
    if not (0 <= shuff <= 1):
        argparser.error("Shuffle Bavior must be between 0 and 1")
    #global PLAYING_RULE
    #PLAYING_RULE = diff
    #global CARD_SHUFFLING_RULE
    #CARD_SHUFFLING_RULE = shuff



    pygame.init()

    # Initialize game and renderer
    game = JaxBlackjack()
    renderer = BlackjackRenderer()
    scaling = 4

    screen = pygame.display.set_mode((160 * scaling, 210 * scaling))
    pygame.display.set_caption("Blackjack")

    jitted_step = jax.jit(game.step)
    jitted_render = jax.jit(renderer.render)
    jitted_reset = jax.jit(game.reset)

    init_obs, state = jitted_reset()

    # Setup game loop
    clock = pygame.time.Clock()
    running = True
    done = False

    while running and not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = Action.UP
        elif keys[pygame.K_s]:
            action = Action.DOWN
        elif keys[pygame.K_SPACE]:
            action = Action.FIRE
        else:
            action = Action.NOOP

        # Update game state
        obs, state, reward, done, info = (jitted_step(state, action))

        # Render
        frame = jitted_render(state)

        aj.update_pygame(screen, frame, scaling, 160, 210)

        # Cap at 60 FPS
        clock.tick(15)

    # If game over, wait before closing
    if done:
        pygame.time.wait(2000)

    pygame.quit()

class BlackjackRenderer(JAXGameRenderer):
    def __init__(self, consts: BlackjackConstants = None):
        super().__init__()
        self.consts = consts or BlackjackConstants()
        (
            self.SPRITE_BACKGROUND,
            self.SPRITE_BJ,
            self.SPRITE_BUST,
            self.SPRITE_DBLE,
            self.SPRITE_HIT,
            self.SPRITE_LOSE,
            self.SPRITE_STAY,
            self.SPRITE_WIN,
            self.SPRITE_TIE,
            self.SPRITE_CARD_EMPTY,
            self.SPRITE_QUESTIONMARK,
            self.SPRITE_CARD_BLACK_2,
            self.SPRITE_CARD_BLACK_3,
            self.SPRITE_CARD_BLACK_4,
            self.SPRITE_CARD_BLACK_5,
            self.SPRITE_CARD_BLACK_6,
            self.SPRITE_CARD_BLACK_7,
            self.SPRITE_CARD_BLACK_8,
            self.SPRITE_CARD_BLACK_9,
            self.SPRITE_CARD_BLACK_10,
            self.SPRITE_CARD_BLACK_J,
            self.SPRITE_CARD_BLACK_K,
            self.SPRITE_CARD_BLACK_Q,
            self.SPRITE_CARD_BLACK_A,
            self.SPRITE_CARD_RED_2,
            self.SPRITE_CARD_RED_3,
            self.SPRITE_CARD_RED_4,
            self.SPRITE_CARD_RED_5,
            self.SPRITE_CARD_RED_6,
            self.SPRITE_CARD_RED_7,
            self.SPRITE_CARD_RED_8,
            self.SPRITE_CARD_RED_9,
            self.SPRITE_CARD_RED_10,
            self.SPRITE_CARD_RED_J,
            self.SPRITE_CARD_RED_K,
            self.SPRITE_CARD_RED_Q,
            self.SPRITE_CARD_RED_A,
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
        frame_bg = aj.get_sprite_frame(self.SPRITE_BACKGROUND, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render questionmark, when Player can select bet amount
        frame_questionmark = aj.get_sprite_frame(self.SPRITE_QUESTIONMARK, 0)
        raster = jax.lax.cond(
            pred = state.state_counter == 1,
            true_fun = lambda: aj.render_at(raster, 37, 59, frame_questionmark),
            false_fun = lambda: raster
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
        frame_hit = aj.get_sprite_frame(self.SPRITE_HIT, 0)
        frame_stay = aj.get_sprite_frame(self.SPRITE_STAY, 0)
        frame_dble = aj.get_sprite_frame(self.SPRITE_DBLE, 0)
        frame_win = aj.get_sprite_frame(self.SPRITE_WIN, 0)
        frame_lose = aj.get_sprite_frame(self.SPRITE_LOSE, 0)
        frame_tie = aj.get_sprite_frame(self.SPRITE_TIE, 0)
        frame_bj = aj.get_sprite_frame(self.SPRITE_BJ, 0)
        frame_bust = aj.get_sprite_frame(self.SPRITE_BUST, 0)

        def get_selected_action_raster(raster, player_action):
            """ Tells whether hit, stay or double is selected, 0 = stay, 1 = double, 2 = hit """
            rasters = jnp.array([
                aj.render_at(raster, 36, 74, frame_stay),
                aj.render_at(raster, 36, 74, frame_dble),
                aj.render_at(raster, 36, 74, frame_hit)
            ])
            return rasters[player_action]


        def get_description_raster(raster, last_round):
            """ Tells whether win, lose, tie, bj, bust or nothing should be selected, 0 = nothing, 1 = win, 2 = lose, 3 = tie, 4= bj, 5 = bust """
            rasters = jnp.array([
                raster,
                aj.render_at(raster, 36, 74, frame_win),
                aj.render_at(raster, 36, 74, frame_lose),
                aj.render_at(raster, 36, 74, frame_tie),
                aj.render_at(raster, 36, 74, frame_bj),
                aj.render_at(raster, 36, 74, frame_bust)
            ])
            return rasters[last_round]

        raster = jax.lax.cond(
            pred = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(state.state_counter == 3, state.state_counter == 4), jnp.logical_or(state.state_counter == 5, state.state_counter == 6)),
                jnp.logical_or(jnp.logical_or(state.state_counter == 7, state.state_counter == 8), state.state_counter == 9)
            ),
            true_fun = lambda: get_selected_action_raster(raster, state.player_action),
            false_fun = lambda: get_description_raster(raster, state.last_round)
        )

        def get_card_sprite(card_number):
            """ Calculates the sprite to the given number """
            card_sprites = jnp.array([
                self.SPRITE_CARD_BLACK_2,
                self.SPRITE_CARD_BLACK_3,
                self.SPRITE_CARD_BLACK_4,
                self.SPRITE_CARD_BLACK_5,
                self.SPRITE_CARD_BLACK_6,
                self.SPRITE_CARD_BLACK_7,
                self.SPRITE_CARD_BLACK_8,
                self.SPRITE_CARD_BLACK_9,
                self.SPRITE_CARD_BLACK_10,
                self.SPRITE_CARD_BLACK_J,
                self.SPRITE_CARD_BLACK_K,
                self.SPRITE_CARD_BLACK_Q,
                self.SPRITE_CARD_BLACK_A,
                self.SPRITE_CARD_RED_2,
                self.SPRITE_CARD_RED_3,
                self.SPRITE_CARD_RED_4,
                self.SPRITE_CARD_RED_5,
                self.SPRITE_CARD_RED_6,
                self.SPRITE_CARD_RED_7,
                self.SPRITE_CARD_RED_8,
                self.SPRITE_CARD_RED_9,
                self.SPRITE_CARD_RED_10,
                self.SPRITE_CARD_RED_J,
                self.SPRITE_CARD_RED_K,
                self.SPRITE_CARD_RED_Q,
                self.SPRITE_CARD_RED_A
            ])
            sprite = card_sprites[jnp.ceil(card_number / 2).astype(jnp.int32) - 1]

            return sprite


        # Render cards of player
        raster = jax.lax.fori_loop(
            lower = 0,
            upper = state.cards_player_counter,
            body_fun = lambda i, val: aj.render_at(val, 44, 88 + i * 19, aj.get_sprite_frame(get_card_sprite(state.cards_player[i]), 0)),
            init_val = raster
        )


        # Render cards of dealer
        raster = jax.lax.cond(
            # If the dealer has only 2 cards, the second card will be hidden
            pred = jnp.logical_and(state.cards_dealer_counter == 2, jnp.logical_not(jnp.logical_or(jnp.logical_or(state.state_counter == 8, state.state_counter == 9), jnp.logical_or(state.state_counter == 10, state.state_counter == 1) ))),
            true_fun = lambda: aj.render_at(aj.render_at(raster, 44, 3, aj.get_sprite_frame(get_card_sprite(state.cards_dealer[0]), 0)), 76, 3, aj.get_sprite_frame(self.SPRITE_CARD_EMPTY, 0)),
            false_fun = lambda: jax.lax.fori_loop(
                lower = 0,
                upper = state.cards_dealer_counter,
                body_fun = lambda i, val: jax.lax.cond(
                    pred = i <= 2,
                    true_fun = lambda: aj.render_at(val, 44 + i * 32, 3, aj.get_sprite_frame(get_card_sprite(state.cards_dealer[i]), 0)),
                    false_fun = lambda: aj.render_at(val, 44 + (i - 3) * 32, 22, aj.get_sprite_frame(get_card_sprite(state.cards_dealer[i]), 0))
                ),
                init_val = raster
            )
        )

        return raster

if __name__ == "__main__":
    main()
