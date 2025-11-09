from dataclasses import dataclass
from jax import numpy as jnp
from jaxatari.core import JAXAtariGame
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import chex

# =============================================================================
# TASK 1: Define Constants and Game State
# =============================================================================

class DunkConstants:
    """
    Holds all static values for the game like screen dimensions, player speeds, colors, etc.
    """
    WINDOW_WIDTH: int = 200
    WINDOW_HEIGHT: int = 240
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER1_COLOR: Tuple[int, int, int] = (200, 72, 72)
    PLAYER2_COLOR: Tuple[int, int, int] = (72, 72, 200)
    BALL_COLOR: Tuple[int, int, int] = (204, 102, 0)
    BALL_SIZE: Tuple[int, int] = (3,3)
    BALL_START: Tuple [int, int] = (100, 70)
    WALL_COLOR: Tuple[int, int, int] = (142, 142, 142)
    FIELD_COLOR: Tuple[int, int, int] = (128, 128, 128)
    JUMP_HEIGHT: int = 5 #adjustable if necessary and more of a placeholder value
    PLAYER_MAX_SPEED: int = 6 #adjustable if necessary and more of a placeholder value
    PLAYER_Y_MIN: int = 50
    PLAYER_Y_MAX: int = 100
    PLAYER_X_MIN: int  = 20
    PLAYER_X_MAX: int = 180
    PLAYER_ROLES: Tuple[int,int] = (0,1) #0 = Offence, 1 = Defence (might be doable with booleans as well)
    BASKET_POSITION: Tuple[int,int] = (100,130)
    BASKET_BUFFER: int = 3 #this should translate to [BASKET_POSITION[0]-buffer:BASKET_POSITION[0]+buffer] being the valid goal area width-wise
    AREA_3_POINT: Tuple[int,int] #We need a way to determine whether a player is in a 3-point area (might be easier to define the two-point area and the rest
                                # will be considered 3-point by process of elimination)


@dataclass
class DunkGameState:
    player1_inside_x: chex.Array
    player1_inside_y: chex.Array
    player1_outside_x: chex.Array
    player1_outside_y: chex.Array
    player2_inside_x: chex.Array
    player2_inside_y: chex.Array
    player2_outside_x: chex.Array
    player2_outside_y: chex.Array

    player1_vel_inside_x: chex.Array
    player1_vel_inside_y: chex.Array
    player1_vel_outside_x: chex.Array
    player1_vel_outside_y: chex.Array
    player2_vel_inside_x: chex.Array
    player2_vel_inside_y: chex.Array
    player2_vel_outside_x: chex.Array
    player2_vel_outside_y: chex.Array

    player1_role: chex.Array
    player2_role: chex.Array

    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    acceleration_counter: chex.Array



class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class DunkObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    ball: EntityPosition
    score_player: jnp.ndarray
    score_enemy: jnp.ndarray


class DunkInfo(NamedTuple):
    time: jnp.ndarray

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

