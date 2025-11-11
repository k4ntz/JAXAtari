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
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.spaces import Space, Discrete, Box, Dict


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Blackjack.
    """
    # Define file lists for groups
    labels = ["stay", "dble", "hit", "win", "lose", "tie", "bj", "bust"]
    label_files = [f"labels/{label}.npy" for label in labels]
    
    numbers = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "K", "Q", "A"]
    black_card_files = [f"cards/black/card_black_{n}.npy" for n in numbers]
    red_card_files = [f"cards/red/card_red_{n}.npy" for n in numbers]
    
    config = (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'card_empty', 'type': 'single', 'file': 'card_empty.npy'},
        {'name': 'questionmark', 'type': 'single', 'file': 'questionmark.npy'},
        
        # Groups
        {'name': 'labels', 'type': 'group', 'files': label_files},
        {'name': 'black_cards', 'type': 'group', 'files': black_card_files},
        {'name': 'red_cards', 'type': 'group', 'files': red_card_files},
        
        # Digits
        {'name': 'player_score_digits', 'type': 'digits', 'pattern': 'digits/white/digit_white_{}.npy'},
        {'name': 'bet_digits', 'type': 'digits', 'pattern': 'digits/black/digit_black_{}.npy'},
    )
    
    return config


class BlackjackConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
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
    # Asset config baked into constants
    ASSET_CONFIG: tuple = _get_default_asset_config()



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


class JaxBlackjack(JaxEnvironment[BlackjackState, BlackjackObservation, BlackjackInfo, BlackjackConstants]):
    def __init__(self, consts: BlackjackConstants = None):
        super().__init__()
        self.consts = consts or BlackjackConstants()
        self.renderer = BlackjackRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN
        ]
        self.obs_size = 5

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_card_value_sum(self, hand: chex.Array):
        """ Calculates the card value of all cards in hand """

        def sum_cards(i, param):
            """ Adds the current card value accept aces to the sum and updates the number of aces """
            csum, number_of_A = param
            value = hand[i]

            # Checks the value for the current card and adds it to csum
            # Aces are ignored
            csum = jnp.where(value != 0, csum + self.consts.CARD_VALUES[(value - 1) % 13], csum)

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

    @partial(jax.jit, static_argnums=(0,))
    def _select_bet_value(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 1: Select bet value with Up/Down and confirm with Fire (and reset displayed cards) """
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
            true_fun=lambda s: self._draw_initial_cards(s, action),
            false_fun=lambda s: s,
            operand=new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _draw_initial_cards(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 2: Sample card for player, dealer, player and dealer (last card for dealer is hidden) """
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
                jnp.logical_or(self._calculate_card_value_sum(op[0]) == 21, self._calculate_card_value_sum(op[1]) == 21),
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
                lambda s: s,
                # Step 3
                lambda s: s,
                # Step 10
                lambda s: self._check_winner(s, action)
            ],
            operand=new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _select_next_action(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 3: Select hit, stay or double with Up/Down and confirm with Fire """
        player_cards_sum = self._calculate_card_value_sum(state.cards_player)

        # Conditions
        # The player has enough points left when loosing with double selected
        has_enough_points_to_double = state.player_bet * 2 > state.player_score
        # If Playing Rule 0 is selected, then the player can select double only if he has a card sum of 10 or 11
        playing_rule_condition = jnp.logical_or(jnp.logical_and(jnp.logical_and(player_cards_sum != 10, player_cards_sum != 11), self.consts.PLAYING_RULE == 0), self.consts.PLAYING_RULE == 1)
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
                lambda s: s,
                # Step 4
                lambda s: self._execute_hit(s, action),
                # Step 5
                lambda s: self._execute_stay(s, action),
                # Step 6
                lambda s: self._execute_double(s, action),
                # Step 10
                lambda s: self._check_winner(s, action)
            ],
            operand=new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _execute_hit(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 4: Execute hit """
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

        return self._check_player_bust(new_state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _execute_stay(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 5: Execute stay """
        # change the next step to state 8
        new_state_counter = jnp.array(8).astype(jnp.int32)
        return self._check_dealer_draw(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _execute_double(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 6: Execute double """
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
        new_state_counter = jnp.where(self._calculate_card_value_sum(new_cards_player) > 21, jnp.array(10).astype(jnp.int32), jnp.array(8).astype(jnp.int32))

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
            true_fun=lambda s: self._check_dealer_draw(s, action),
            false_fun=lambda s: self._check_winner(s, action),
            operand=new_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_player_bust(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 7: Check if player has 21 or more """
        player_cards_sum = self._calculate_card_value_sum(state.cards_player)

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
                lambda s: self._check_dealer_draw(s, action),
                # Step 10
                lambda s: self._check_winner(s, action),
                # Step 3
                lambda s: s
            ],
            operand=new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_dealer_draw(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 8: Check if dealer should take a card """
        card_sum = self._calculate_card_value_sum(state.cards_dealer)
        condition = jnp.array([
            jnp.logical_or(card_sum > 17, jnp.logical_and(card_sum >= 17, jnp.any(jnp.isin(state.cards_dealer, jnp.array([25, 26, 51, 52]))))),
            card_sum > 16
        ], dtype=bool)

        new_state_counter = jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_not(state.cards_dealer_counter == 6), jnp.logical_not(condition[self.consts.PLAYING_RULE])),
            true_fun=lambda: jnp.array(9).astype(jnp.int32),
            false_fun=lambda: jnp.array(10).astype(jnp.int32)
        )

        return jax.lax.cond(
            pred=new_state_counter == 10,
            true_fun=lambda s: self._check_winner(s, action),
            false_fun=lambda s: self._execute_dealer_draw(s, action),
            operand=state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _execute_dealer_draw(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 9: Give dealer a card """
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

    @partial(jax.jit, static_argnums=(0,))
    def _check_winner(self, state: BlackjackState, action: chex.Array) -> BlackjackState:
        """ Step 10: Check winner """
        hand_player = self._calculate_card_value_sum(state.cards_player)
        hand_dealer = self._calculate_card_value_sum(state.cards_dealer)

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
                pred=self.consts.PLAYING_RULE == 0,
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
                jnp.logical_or(hand_dealer > 21, jnp.logical_and(self.consts.PLAYING_RULE == 1, state.cards_player_counter >= 6)), # dealer has more than 21 OR [rule PBJR] player has (more than) 6 cards
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
        condition = jnp.logical_or(self.consts.CARD_SHUFFLING_RULE == 0, jnp.logical_and(self.consts.CARD_SHUFFLING_RULE == 1, state.card_permutation_counter >= 34))

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
            true_fun= lambda s: jax.lax.switch(
                s.state_counter - 1,
                [
                    lambda op: self._select_bet_value(op[0], op[1]),
                    lambda op: self._draw_initial_cards(op[0], op[1]),
                    lambda op: self._select_next_action(op[0], op[1]),
                    lambda op: self._execute_hit(op[0], op[1]),
                    lambda op: self._execute_stay(op[0], op[1]),
                    lambda op: self._execute_double(op[0], op[1]),
                    lambda op: self._check_player_bust(op[0], op[1]),
                    lambda op: self._check_dealer_draw(op[0], op[1]),
                    lambda op: self._execute_dealer_draw(op[0], op[1]),
                    lambda op: self._check_winner(op[0], op[1])
                ],
                (s, action)
            ),
            false_fun= lambda s: BlackjackState(
                player_score=s.player_score,
                player_bet=s.player_bet,
                player_action=s.player_action,
                cards_player=s.cards_player,
                cards_player_counter=s.cards_player_counter,
                cards_dealer=s.cards_dealer,
                cards_dealer_counter=s.cards_dealer_counter,
                step_counter=s.step_counter + 1,
                state_counter=s.state_counter,
                cards_permutation=s.cards_permutation,
                card_permutation_counter=s.card_permutation_counter,
                key=s.key,
                subkey=s.subkey,
                last_round=s.last_round,
                skip_step=s.skip_step
            ),
            operand=state
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BlackjackState, state: BlackjackState):
        return state.player_score - previous_state.player_score

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
        player_sum = self._calculate_card_value_sum(state.cards_player)
        # Assuming dealer's first card is the upcard for observation
        dealer_sum = jax.lax.cond(
            pred=state.cards_dealer_counter == 2,
            true_fun=lambda s: self._calculate_card_value_sum(jnp.concatenate([s.cards_dealer[:1], s.cards_dealer[1:]])),
            false_fun=lambda s: self._calculate_card_value_sum(s.cards_dealer),
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
    def _get_info(self, state: BlackjackState) -> BlackjackInfo:
        return BlackjackInfo(state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BlackjackState) -> bool:
        player_has_no_money = state.player_score <= 0
        player_wins = state.player_score >= 1000
        return jnp.logical_or(player_has_no_money, player_wins)


class BlackjackRenderer(JAXGameRenderer):
    def __init__(self, consts: BlackjackConstants = None):
        super().__init__()
        self.consts = consts or BlackjackConstants()
        # 1. Configure the renderer
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 2. Define asset path
        sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites/blackjack")
        # 3. Load all assets using the manifest from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BlackjackState):
        """ Responsible for the graphical representation of the game """
        # Create 2D ID raster from the background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render questionmark, when Player can select bet amount
        raster = jax.lax.cond(
            pred = state.state_counter == 1,
            true_fun = lambda r: self.jr.render_at(
                r, 37, 59, 
                self.SHAPE_MASKS['questionmark'], 
                flip_offset=self.FLIP_OFFSETS['questionmark']
            ),
            false_fun = lambda r: r,
            operand=raster
        )

        # Render player_score
        # 1. Get digit array (always 4 digits)
        player_score_digits = self.jr.int_to_digits(state.player_score, max_digits=4)

        # 2. Determine parameters for player score rendering
        player_score_conditions = jnp.array([
            state.player_score < 10,
            jnp.logical_and(state.player_score >= 10, state.player_score < 100),
            jnp.logical_and(state.player_score >= 100, state.player_score < 1000),
            state.player_score >= 1000
        ], dtype=bool)
        player_score_start_index = jnp.select(player_score_conditions, jnp.array([3, 2, 1, 0]))
        player_score_num_to_render = jnp.select(player_score_conditions, jnp.array([1, 2, 3, 4]))

        # 3. Render player score using the selective renderer
        raster = self.jr.render_label_selective(
            raster, 41, 44,
            player_score_digits, 
            self.SHAPE_MASKS['player_score_digits'],
            player_score_start_index, 
            player_score_num_to_render,
            spacing=5,
            max_digits_to_render=4
        )

        # Render player_bet
        # 1. Get digit array (always 2 digits)
        player_bet_digits = self.jr.int_to_digits(state.player_bet, max_digits=2)

        # 2. Determine parameters for player score rendering
        is_player_bet_single_digit = state.player_bet < 10
        player_bet_start_index = jax.lax.select(is_player_bet_single_digit, 1, 0)
        player_bet_num_to_render = jax.lax.select(is_player_bet_single_digit, 1, 2)

        # 3. Render player bet using the selective renderer
        raster = self.jr.render_label_selective(
            raster, 45, 59,
            player_bet_digits, 
            self.SHAPE_MASKS['bet_digits'],
            player_bet_start_index, 
            player_bet_num_to_render,
            spacing=5,
            max_digits_to_render=2
        )

        # Render player action in step 3, 4, 5, 6, 7, 8, 9
        def render_label_at(r, name_idx):
            return self.jr.render_at(
                r, 36, 74, 
                self.SHAPE_MASKS['labels'][name_idx], 
                flip_offset=self.FLIP_OFFSETS['labels']
            )

        raster = jnp.select(
            condlist=[
                jnp.logical_and(state.state_counter >= 3, state.state_counter <= 9),
                jnp.logical_and(state.state_counter != 2, state.last_round != 0)
            ],
            choicelist=[
                render_label_at(raster, state.player_action),
                render_label_at(raster, 3 + state.last_round - 1)
            ],
            default=raster
        )

        def get_card_sprite_and_offset(card_number):
            """ Calculates the sprite mask and flip offset for a given card number """
            is_black = jnp.less_equal(card_number / 13, 2)
            
            def get_black(cn):
                mask = self.SHAPE_MASKS['black_cards'][(cn - 1) % 13]
                offset = self.FLIP_OFFSETS['black_cards']
                return mask, offset
                
            def get_red(cn):
                mask = self.SHAPE_MASKS['red_cards'][(cn - 1) % 13]
                offset = self.FLIP_OFFSETS['red_cards']
                return mask, offset

            return jax.lax.cond(
                is_black,
                get_black,
                get_red,
                operand=card_number
            )

        # Render cards of player
        def render_player_card_loop(i, val_raster):
            sprite, offset = get_card_sprite_and_offset(state.cards_player[i])
            return self.jr.render_at(val_raster, 44, 88 + i * 19, sprite, flip_offset=offset)
        
        raster = jax.lax.fori_loop(
            lower = 0,
            upper = state.cards_player_counter,
            body_fun = render_player_card_loop,
            init_val = raster
        )

        # Render cards of the dealer
        def render_dealer_choicelist_0(r):
            sprite0, offset0 = get_card_sprite_and_offset(state.cards_dealer[0])
            sprite1, offset1 = get_card_sprite_and_offset(state.cards_dealer[1])
            r = self.jr.render_at(r, 44, 3, sprite0, flip_offset=offset0)
            r = self.jr.render_at(r, 76, 3, sprite1, flip_offset=offset1)
            return r

        def render_dealer_choicelist_1(r):
            sprite0, offset0 = get_card_sprite_and_offset(state.cards_dealer[0])
            r = self.jr.render_at(r, 44, 3, sprite0, flip_offset=offset0)
            r = self.jr.render_at(
                r, 76, 3, 
                self.SHAPE_MASKS['card_empty'], 
                flip_offset=self.FLIP_OFFSETS['card_empty']
            )
            return r

        def render_dealer_default_loop(i, val_raster):
            sprite, offset = get_card_sprite_and_offset(state.cards_dealer[i])
            return jax.lax.cond(
                pred=i <= 2,
                true_fun=lambda: self.jr.render_at(val_raster, 44 + i * 32, 3, sprite, flip_offset=offset),
                false_fun=lambda: self.jr.render_at(val_raster, 44 + (i - 3) * 32, 22, sprite, flip_offset=offset)
            )

        raster = jnp.select(
            condlist=[
                jnp.logical_and(state.cards_dealer_counter == 2, state.skip_step - state.step_counter >= 40),
                jnp.logical_and(state.cards_dealer_counter == 2, jnp.logical_not(jnp.isin(state.state_counter, jnp.array([8, 9, 10, 1], dtype=jnp.int32))))
            ],
            choicelist=[
                render_dealer_choicelist_0(raster),
                render_dealer_choicelist_1(raster),
            ],
            default=jax.lax.fori_loop(
                    lower=0,
                    upper=state.cards_dealer_counter,
                    body_fun=render_dealer_default_loop,
                    init_val=raster
                ),
        )

        # Final conversion from ID raster to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)