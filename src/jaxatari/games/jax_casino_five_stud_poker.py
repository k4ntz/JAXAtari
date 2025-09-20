#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX Casino Five Stud Poker
#
# Simulates the Atari Casino Five Stud Poker game
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember

from typing import NamedTuple, Tuple
from functools import partial

import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JAXAtariAction as Action, JaxEnvironment
from jaxatari.spaces import Space, Discrete, Box, Dict
from jax_casino_renderer import CasinoRenderer


@jax.jit
def evaluate_hand(hand):
    """
    Evaluates a 5-card poker hand.
    Returns a tuple: (hand_rank, tie_breaker_ranks)
    hand_rank: 9 (Royal Flush) down to 0 (High Card)
    tie_breaker_ranks: Sorted ranks to break ties.
    """
    # A value of -2 indicates an empty slot, so we must handle this.
    valid_cards = hand > 0
    cards_0_indexed = jnp.where(valid_cards, hand - 1, -1)

    ranks = jnp.sort(jnp.where(valid_cards, cards_0_indexed % 13, -1))
    suits = jnp.where(valid_cards, cards_0_indexed // 13, -1)

    # Calculations should only happen if there are 5 cards.
    is_full_hand = jnp.sum(valid_cards) == 5

    is_flush = jnp.all(suits[0] == suits) & is_full_hand
    
    # Check for straights, including Ace-low (A-2-3-4-5)
    is_straight_ace_low = jnp.array_equal(ranks, jnp.array([0, 1, 2, 3, 12]))
    # Need to check for unique ranks to avoid hands like [2,2,3,4,5] being a straight
    is_straight_normal = (ranks[4] - ranks[0] == 4) & (jnp.unique(ranks, size=5).size == 5)
    is_straight = (is_straight_normal | is_straight_ace_low) & is_full_hand

    # Tie-breaking ranks for straights/flushes (high card)
    # For Ace-low straight, the high card is 5 (rank 3), but Ace is still used for tie-breaking if needed
    straight_flush_ranks = jax.lax.cond(is_straight_ace_low,
                                  lambda _: jnp.array([3, 2, 1, 0, 12]), # A-5 -> 5 high, but A is rank 12
                                  lambda r: r[::-1],
                                  ranks) # Normal order

    # Check for pairs, three of a kind, etc.
    unique_ranks, counts = jnp.unique(ranks, return_counts=True, size=5, fill_value=-1)
    
    # Sort unique ranks by count (desc), then by rank (desc) for tie-breaking
    sorted_indices = jnp.lexsort((-unique_ranks, -counts))
    pair_tie_breaker_ranks = unique_ranks[sorted_indices]

    counts = jnp.sort(counts)[::-1]

    is_four_of_a_kind = (counts[0] == 4) & is_full_hand
    is_full_house = ((counts[0] == 3) & (counts[1] == 2)) & is_full_hand
    is_three_of_a_kind = (counts[0] == 3) & is_full_hand
    is_two_pair = ((counts[0] == 2) & (counts[1] == 2)) & is_full_hand
    is_one_pair = (counts[0] == 2) & is_full_hand

    is_straight_flush = is_straight & is_flush
    # Royal flush is the highest straight flush (A-K-Q-J-10)
    is_royal_flush = is_straight_flush & (ranks[4] == 12) & (ranks[0] == 8) # 10,J,Q,K,A

    # Assign hand rank
    hand_rank = jnp.select(
        [is_royal_flush, is_straight_flush, is_four_of_a_kind, is_full_house, is_flush, is_straight, is_three_of_a_kind, is_two_pair, is_one_pair],
        [9, 8, 7, 6, 5, 4, 3, 2, 1],
        default=0 # High card
    )
    
    # Select the correct tie-breaker array
    final_tie_breaker = jax.lax.cond(is_straight | is_flush, 
                                     lambda op: op[0], 
                                     lambda op: op[1],
                                     (straight_flush_ranks, pair_tie_breaker_ranks))

    # If not a full hand, rank is -1
    return jax.lax.cond(is_full_hand, 
                        lambda op: (op[0], op[1]), 
                        lambda _: (-1, jnp.full(5, -1, dtype=jnp.int32)),
                        (hand_rank, final_tie_breaker))

@jax.jit
def betting_step(state, action, min_bet, max_bet):
    """Handles player betting."""
    
    # Player presses FIRE
    is_fire = action == Action.FIRE
    
    # The minimum bet for this action is the bet from the previous round.
    # For the first betting round (ante), it's the global MIN_BET.
    current_min_bet = jax.lax.cond(state.state_counter == 0, lambda _: min_bet, lambda s: s.total_bet, state)
    
    # Fold by pressing FIRE when the current bet equals the previous round's total bet.
    is_fold = is_fire & (state.round_bet == state.total_bet) & (state.state_counter > 0)
    
    # A normal bet is when FIRE is pressed with a bet > previous total bet
    is_normal_bet = is_fire & (state.round_bet > state.total_bet)

    # Ante bet at the start of the round
    is_ante_bet = is_fire & (state.state_counter == 0)

    # The maximum bet for this round is the previous total bet plus the per-round max.
    current_max_bet = state.total_bet + max_bet

    # Increase bet
    new_round_bet = jax.lax.cond(
        (action == Action.UP) & (state.round_bet < current_max_bet),
        lambda r_bet: r_bet + 10,
        lambda r_bet: r_bet,
        state.round_bet
    )
    # Decrease bet (but not below the total bet from the previous round)
    new_round_bet = jax.lax.cond(
        (action == Action.DOWN) & (new_round_bet > current_min_bet),
        lambda r_bet: r_bet - 10,
        lambda r_bet: r_bet,
        new_round_bet
    )

    # On a valid bet (normal or ante), the total_bet becomes the current round_bet.
    new_total_bet = jax.lax.cond(is_normal_bet | is_ante_bet, 
                                 lambda op: op[0], 
                                 lambda op: op[1], 
                                 (new_round_bet, state.total_bet))
    
    # Score is updated at the end of the round in payout_step
    new_player_score = state.player_score
    
    # Advance state on a normal bet, or jump to showdown on a fold
    next_state_counter = jax.lax.cond(is_normal_bet | is_ante_bet, 
                                       lambda sc: sc + 1, 
                                       lambda sc: jax.lax.cond(is_fold, 
                                                            lambda _: 9, # Jump to showdown
                                                            lambda s_sc: s_sc,
                                                            sc),
                                       state.state_counter)

    player_folded = jax.lax.cond(is_fold, lambda _: 1, lambda pf: pf, state.player_folded)

    # After a bet is placed, the next round's bet display should start at the new total
    # The default bet for the next round should be the new total plus the amount just bet.
    last_bet_amount = new_total_bet - state.total_bet
    next_round_default_bet = new_total_bet + last_bet_amount
    # Ensure the default bet does not exceed the maximum for the next round
    next_round_max = new_total_bet + max_bet
    next_round_default_bet = jnp.minimum(next_round_default_bet, next_round_max)

    new_round_bet = jax.lax.cond(is_normal_bet | is_ante_bet, lambda _: next_round_default_bet, lambda _: new_round_bet, None)

    return state._replace(
        round_bet=new_round_bet,
        total_bet=new_total_bet,
        player_score=new_player_score,
        state_counter=next_state_counter,
        player_folded=player_folded,
    )

@partial(jax.jit, static_argnames='num_cards_each')
def deal_step(state, num_cards_each):
    """Deals cards to player and dealer."""

    # dynamically choose cards for player and dealer by slicing deck from state.deck_idx to state.deck_idx + num_cards_each (1 or 2 cards based on the step)
    player_new_cards = jax.lax.dynamic_slice(state.deck, (state.deck_idx,), (num_cards_each,))
    dealer_new_cards = jax.lax.dynamic_slice(state.deck, (state.deck_idx + num_cards_each,), (num_cards_each,))

    # Deals cards to the next available slot.
    # state_counter = 1 -> idx=0. state_counter = 3 -> idx=2. state_counter = 5 -> idx=3 etc.
    card_start_idx = (state.state_counter - 1) // 2 + (state.state_counter > 1)
    
    new_player_cards = jax.lax.dynamic_update_slice(state.player_cards, player_new_cards, (card_start_idx,))
    new_dealer_cards = jax.lax.dynamic_update_slice(state.dealer_cards, dealer_new_cards, (card_start_idx,))

    return state._replace(
        player_cards=new_player_cards,
        dealer_cards=new_dealer_cards,
        deck_idx=state.deck_idx + num_cards_each * 2,
        state_counter=state.state_counter + 1
    )

@jax.jit
def showdown_step(state):
    """Evaluates hands and determines the winner."""
    player_rank, player_tie_breaker = evaluate_hand(state.player_cards)
    dealer_rank, dealer_tie_breaker = evaluate_hand(state.dealer_cards)

    # Compare ranks
    winner = jax.lax.cond(player_rank > dealer_rank, lambda _: 1, # Player wins
              lambda op: jax.lax.cond(op[1] > op[0], lambda _: 2, # Dealer wins
              # Ranks are equal, check tie-breakers
              lambda tie_breakers: compare_tie_breakers(tie_breakers[0], tie_breakers[1]),(op[2], op[3]),
             ), (player_rank, dealer_rank, player_tie_breaker, dealer_tie_breaker))
    
    # Player loses automatically if they folded
    result = jax.lax.cond(state.player_folded == 1, lambda _: 2, lambda w: w, winner)

    return state._replace(state_counter=10, last_round_result=result)


@jax.jit
def compare_tie_breakers(player_tie_breaker, dealer_tie_breaker):
    """
    Compares tie-breaker ranks lexicographically.
    Returns: 1 if player wins, 2 if dealer wins, 3 if it's a tie.
    """
    # Assumes tie_breaker arrays are sorted in descending order of importance.
    def loop_body(i, winner_so_far):
        # If a winner has been found in a previous iteration, don't change it.
        return jax.lax.cond(
            winner_so_far != 3,
            lambda w: w,
            # If still a tie, compare the current rank.
            lambda _: jax.lax.cond(
                player_tie_breaker[i] > dealer_tie_breaker[i],
                lambda _: 1, # Player wins
                lambda op: jax.lax.cond(
                    op[1][i] > op[0][i],
                    lambda _: 2, # Dealer wins
                    lambda _: 3,  # Still a tie, continue to next rank.
                    op
                ),
                (player_tie_breaker, dealer_tie_breaker)
            ),
            winner_so_far
        )
    # Iterate through the tie-breaker ranks. Start with a tie (3).
    winner = jax.lax.fori_loop(0, player_tie_breaker.shape[0], loop_body, 3)
    return winner

@jax.jit
def payout_step(state):
    """Calculates the payout based on the round result."""
    # Player wins: score + total_bet (winnings)
    # Player loses/folds: score - total_bet (loss)
    # Tie: score (bets are returned)
    new_score = jax.lax.cond(state.last_round_result == 1, # Player won
                             lambda s: s.player_score + s.total_bet,
                             lambda s: jax.lax.cond(s.last_round_result == 3, # Tie
                                                  lambda st: st.player_score,
                                                  lambda st: st.player_score - st.total_bet, # Player lost or folded
                                                  s),
                             state)
    return state._replace(player_score=new_score, state_counter=11)

@partial(jax.jit, static_argnames=('num_cards_in_deck', 'min_bet', 'hand_size'))
def end_round_step(state, action, num_cards_in_deck, min_bet, hand_size):
    """Waits for FIRE action to start a new round."""
    def _start_new_round(st):
        key, subkey = jax.random.split(st.key)
        new_deck = jax.random.permutation(subkey, jnp.arange(1, num_cards_in_deck + 1))
        return st._replace(
            key=key,
            step_counter=jnp.array(0, dtype=jnp.int32),
            state_counter=jnp.array(0, dtype=jnp.int32),
            round_bet=min_bet,
            total_bet=jnp.array(0, dtype=jnp.int32),
            player_cards=jnp.full((hand_size,), -2, dtype=jnp.int32),
            dealer_cards=jnp.full((hand_size,), -2, dtype=jnp.int32),
            deck=new_deck,
            deck_idx=jnp.array(0, dtype=jnp.int32),
            player_folded=jnp.array(0, dtype=jnp.int32),
            last_round_result=jnp.array(0, dtype=jnp.int32),
        )

    return jax.lax.cond(action == Action.FIRE,
                        _start_new_round,
                        lambda s: s,
                        state)

class CasinoFiveStudPokerConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    NUM_PLAYERS = 1  # Versus computer dealer
    INITIAL_PLAYER_SCORE = 1000
    MIN_BET = 10
    MAX_BET = 100
    NUM_CARDS_IN_DECK = 52
    HAND_SIZE = 5
    P1_DIFFICULTY: str = 'b'
    P2_DIFFICULTY: str = 'b'
    BANK_LIMIT = 10000

class CasinoFiveStudPokerState(NamedTuple):
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
    round_bet: jnp.ndarray # Current bet being adjusted by player
    total_bet: jnp.ndarray # Total committed bet for the round
    

    player_cards: jnp.ndarray
    dealer_cards: jnp.ndarray

    deck: jnp.ndarray
    deck_idx: jnp.ndarray

    dealer_card_face_down: jnp.ndarray
    player_card_face_down: jnp.ndarray
    
    player_folded: jnp.ndarray # 0 for not folded, 1 for folded
    last_round_result: jnp.ndarray # 0: none, 1: win, 2: loss, 3: tie

class CasinoFiveStudPokerObservation(NamedTuple):
    player_score: jnp.ndarray
    total_bet: jnp.ndarray
    round_bet: jnp.ndarray
    player_cards: jnp.ndarray
    dealer_cards: jnp.ndarray
    state_counter: jnp.ndarray

class CasinoFiveStudPokerInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: jnp.ndarray

class JaxCasinoFiveStudPoker(JaxEnvironment[CasinoFiveStudPokerState, CasinoFiveStudPokerObservation, CasinoFiveStudPokerInfo, CasinoFiveStudPokerConstants]):
    def __init__(self, consts: CasinoFiveStudPokerConstants = None, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN
        ]
        self.consts = consts or CasinoFiveStudPokerConstants()
        self.renderer = CasinoRenderer()
        self.obs_size = 5

    def reset(self, key=None) -> Tuple[CasinoFiveStudPokerObservation, CasinoFiveStudPokerState]:
        if key is None:
            key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)
        deck = jnp.arange(1, self.consts.NUM_CARDS_IN_DECK + 1)
        deck = jax.random.permutation(subkey, deck)

        state = CasinoFiveStudPokerState(
            key=key,
            step_counter=jnp.array(0, dtype=jnp.int32),
            state_counter=jnp.array(0, dtype=jnp.int32),
            player_score=jnp.array(self.consts.INITIAL_PLAYER_SCORE, dtype=jnp.int32),
            round_bet=jnp.array(self.consts.MIN_BET, dtype=jnp.int32),
            total_bet=jnp.array(0, dtype=jnp.int32),
            player_cards=jnp.full((self.consts.HAND_SIZE,), -2, dtype=jnp.int32),
            dealer_cards=jnp.full((self.consts.HAND_SIZE,), -2, dtype=jnp.int32),
            deck=deck,
            deck_idx=jnp.array(0, dtype=jnp.int32),
            dealer_card_face_down=jnp.array(1 if self.consts.P1_DIFFICULTY == 'a' else 0, dtype=jnp.int32),
            player_card_face_down=jnp.array(1 if self.consts.P2_DIFFICULTY == 'a' else 0, dtype=jnp.int32),
            player_folded=jnp.array(0, dtype=jnp.int32),
            last_round_result=jnp.array(0, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CasinoFiveStudPokerState, action: chex.Array) -> Tuple[CasinoFiveStudPokerObservation, CasinoFiveStudPokerState, float, bool, CasinoFiveStudPokerInfo]:
        previous_state = state
        new_state = state._replace(step_counter=state.step_counter + 1)

        # Game State Machine:
        # 0, 2, 4, 6, 8: Betting rounds (ante, after 2nd, 3rd, 4th, 5th card)
        # 1: Deal initial 2 cards
        # 3, 5, 7: Deal 1 card each
        # 9: Showdown (evaluate hands)
        # 10: Payout
        # 11: End of round, wait for FIRE to restart
        
        # Betting states
        is_betting_state = (new_state.state_counter % 2 == 0) & (new_state.state_counter < 9)
        new_state = jax.lax.cond(is_betting_state,
                             lambda s: betting_step(s, action, self.consts.MIN_BET, self.consts.MAX_BET),
                             lambda s: s,
                             new_state)

        # Dealing states
        new_state = jax.lax.cond(new_state.state_counter == 1, lambda s: deal_step(s, 2), lambda s: s, new_state)
        is_single_deal = (new_state.state_counter == 3) | (new_state.state_counter == 5) | (new_state.state_counter == 7)
        new_state = jax.lax.cond(is_single_deal, lambda s: deal_step(s, 1), lambda s: s, new_state)

        # Showdown, Payout, and End of Round
        new_state = jax.lax.cond(new_state.state_counter == 9, lambda s: showdown_step(s), lambda s: s, new_state)
        new_state = jax.lax.cond(new_state.state_counter == 10, lambda s: payout_step(s), lambda s: s, new_state)
        new_state = jax.lax.cond(new_state.state_counter == 11, lambda s: end_round_step(s, action, self.consts.NUM_CARDS_IN_DECK, self.consts.MIN_BET, self.consts.HAND_SIZE), lambda s: s, new_state)

        reward = self._get_reward(previous_state, new_state)
        done = self._get_done(new_state)
        obs = self._get_observation(new_state)
        all_rewards = self._get_all_reward(previous_state, new_state)
        info = self._get_info(new_state, all_rewards)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: CasinoFiveStudPokerState, state: CasinoFiveStudPokerState):
        # Reward is the change in player score, given only at the end of a round (payout step).
        is_payout_step = state.state_counter == 11
        return jax.lax.cond(is_payout_step,
                            lambda op: (op[1].player_score - op[0].player_score).astype(jnp.float32),
                            lambda _: 0.0,
                            (previous_state, state))

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: CasinoFiveStudPokerState, state: CasinoFiveStudPokerState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state)
             for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CasinoFiveStudPokerState) -> jnp.ndarray:
        # Create card matrix for renderer
        # Game logic: -2=empty, -1=face down, 1-52=cards
        # Renderer: 0=empty, -1=face down, 1-52=cards
        # We need to convert -2 to 0 for the renderer.
        card_matrix = jnp.full((6, 5), 0, dtype=jnp.int32)
        dealer_cards_for_render = jnp.where(state.dealer_cards == -2, 0, state.dealer_cards)
        player_cards_for_render = jnp.where(state.player_cards == -2, 0, state.player_cards)
        card_matrix = card_matrix.at[0, :].set(dealer_cards_for_render)
        card_matrix = card_matrix.at[1, :].set(player_cards_for_render)

        # Determine label based on game state
        # -1: None, 0: Bet, 4: Fold, 7: Lose, 9: Push/Tie, 12: Win
        label_map = jnp.array([-1, 12, 7, 9]) # 0:none, 1:win, 2:loss, 3:tie
        result_label = label_map[state.last_round_result]

        is_betting_state = (state.state_counter % 2 == 0) & (state.state_counter < 9)
        
        # The label should not be visible in the first round's betting phase
        is_first_round_ante = (state.state_counter == 0) & (state.last_round_result == 0)

        label = jnp.select(
            [
                is_first_round_ante,
                state.player_folded == 1,
                (state.last_round_result != 0) & (state.state_counter == 11),
                is_betting_state & (state.round_bet > state.total_bet) & (state.state_counter > 0),
                is_betting_state & (state.round_bet == state.total_bet) & (state.state_counter > 0),
                is_betting_state & (state.state_counter == 0) # Ante round
            ],
            [
                -1, # No label
                4, # Fold
                result_label,
                0, # Bet
                4, # Fold (Call is shown as Fold)
                0, # Bet
            ],
            default=-1
        ).astype(jnp.int32)

        # Determine char based on game state
        # -1: None, 0: i, 1: ?
        char = jnp.select(
            [
                is_betting_state & (state.state_counter == 0), # Initial bet
                is_betting_state # Other bets
            ],
            [
                0, # i
                1  # ?
            ],
            default=-1
        ).astype(jnp.int32)

        return self.renderer.render(
            card_matrix,
            state.player_score,
            self.consts.WIDTH,
            self.consts.HEIGHT,
            char=char,
            player_main_bet=state.round_bet,
            label_main=label,
            blinking_card=jnp.array([-1, -1], dtype=jnp.int32)
        )

    def action_space(self) -> Space:
        return Discrete(len(self.action_set))

    def image_space(self) -> Space:
        return Box(0, 255, (self.consts.HEIGHT, self.consts.WIDTH, 3), jnp.uint8)

    def observation_space(self) -> Space:
        return Dict(
            {
                "player_score": Box(0, 10000, (), jnp.int32),
                "total_bet": Box(0, 500, (), jnp.int32),
                "round_bet": Box(0, 500, (), jnp.int32),
                # Cards are 1-52. -1 is face-down, -2 is empty slot.
                "player_cards": Box(-2, 52, (self.consts.HAND_SIZE,), jnp.int32),
                "dealer_cards": Box(-2, 52, (self.consts.HAND_SIZE,), jnp.int32),
                "state_counter": Box(0, 11, (), jnp.int32),
            }
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CasinoFiveStudPokerObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_score.flatten(),
            obs.total_bet.flatten(),
            obs.round_bet.flatten(),
            obs.player_cards.flatten(),
            obs.dealer_cards.flatten(),
            obs.state_counter.flatten(),
        ]).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: CasinoFiveStudPokerState):
        # Hide dealer's first card if difficulty is 'a' and it's before showdown
        dealer_cards_obs = jax.lax.cond(
            (state.dealer_card_face_down == 1) & (state.state_counter < 9),
            lambda s: s.dealer_cards.at[0].set(-1), # Use -1 to indicate a face-down card
            lambda s: s.dealer_cards,
            state
        )
        # Hide player's first card if difficulty is 'a' and it's before showdown
        player_cards_obs = jax.lax.cond(
            (state.player_card_face_down == 1) & (state.state_counter < 9),
            lambda s: s.player_cards.at[0].set(-1),
            lambda s: s.player_cards,
            state
        )

        return CasinoFiveStudPokerObservation (
            player_score=state.player_score.astype(jnp.int32),
            total_bet=state.total_bet.astype(jnp.int32),
            round_bet=state.round_bet.astype(jnp.int32),
            player_cards=player_cards_obs.astype(jnp.int32),
            dealer_cards=dealer_cards_obs.astype(jnp.int32),
            state_counter=state.state_counter.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CasinoFiveStudPokerState, all_rewards: jnp.ndarray = None) -> CasinoFiveStudPokerInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1)
        return CasinoFiveStudPokerInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: CasinoFiveStudPokerState) -> bool:
        is_broke = state.player_score < self.consts.MIN_BET
        broke_the_bank = state.player_score >= self.consts.BANK_LIMIT
        return is_broke | broke_the_bank