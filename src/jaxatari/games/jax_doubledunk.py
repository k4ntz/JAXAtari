from dataclasses import dataclass
from jax import numpy as jnp
from jaxatari.core import JAXAtariGame 

# =============================================================================
# TASK 1: Define Constants and Game State
# =============================================================================

class Constants:
    """
    Holds all static values for the game like screen dimensions, player speeds, colors, etc.
    """
    pass

@dataclass
class GameState:
    """
    Holds the entire dynamic state of the game.
    """
    pass

class DoubleDunk(JAXAtariGame):
    
    def __init__(self):
        """
        Initialize the game environment.
        """
        self.constants = Constants()

    # =========================================================================
    # TASK 2: Implement Init and Reset
    # =========================================================================
    
    def _init_state(self) -> GameState:
        """
        Creates the very first state of the game.
        Use values from self.constants.
        """
        pass

    def _reset_state(self, state: GameState) -> GameState:
        """
        Resets the state after a point is scored or for a new round.
        """
        pass

    # =========================================================================
    # TASK 3: Implement the Step Function
    # =========================================================================
    def step(self, state: GameState, action: int) -> GameState:
        """
        Takes an action in the game and returns the new game state.
        """
        # Pseudocode for step function logic:
        # 1. Process Player Actions:
        #    - Interpret the 'action' (joystick/button input) based on the current game context:
        #      - If in "Play Selection" mode: Update offensive/defensive play for the team.
        #      - If on OFFENSE:
        #        - Button press: Advance current selected play (pass, set pick).
        #        - Joystick-pull-back + button: Initiate manual jump shot.
        #      - If on DEFENSE:
        #        - Button press near ball carrier: Attempt a steal.
        #        - Jump during opponent's shot: Attempt a block.
        #
        # 2. Update Player Positions (Movement Logic):
        #    - Apply movement velocity to controlled players based on action.
        #    - Apply gravity to players who are jumping.
        #    - Handle court boundary collisions for players.
        #    - Implement AI movement for uncontrolled players (teammates/opponents) based on selected plays.
        #
        # 3. Update Ball Physics (Movement Logic):
        #    - If a player possesses the ball, update ball's position relative to the player.
        #    - If the ball is free (shot, pass, rebound):
        #      - Apply its current velocity and gravity to its position.
        #      - Handle bounces off the floor, backboard, and rim (reverse velocity, apply friction).
        #
        # 4. Handle Collisions & Interactions:
        #    - Player-ball collision:
        #      - If steal attempted: Check for success and change ball ownership.
        #      - If ball loose: Grant ownership to player who touched it.
        #    - Player-player collision: Handle picks, potential fouls.
        #
        # 5. Handle Scoring:
        #    - Check if the ball has successfully passed through a hoop.
        #    - Determine points (2 or 3) based on shot origin.
        #    - Update scores in the game state.
        #    - If scored, trigger a game reset for the next play (e.g., by returning _reset_state(state)).
        #
        # 6. Update Clocks & Check Violations:
        #    - Decrement the main game clock.
        #    - If offensive team has ball:
        #      - Decrement 10-second shot clock; trigger turnover if it expires.
        #      - Monitor 3-second lane violation for offensive players; trigger turnover if violated.
        #
        # 7. Determine Reward & Game Over:
        #    - Calculate and set the 'reward' for the current step (e.g., +1 for scoring, -1 for turnover).
        #    - Check for game-ending conditions:
        #      - If main game clock reaches zero.
        #      - If a team reaches the score limit.
        #    - Set the 'done' flag in the game state to True if the game is over.
        #
        # 8. Return the fully updated GameState object.
        pass

