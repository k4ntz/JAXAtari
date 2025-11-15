from dataclasses import dataclass
from jax import numpy as jnp
from jaxatari.core import JAXAtariGame
from typing import NamedTuple, Tuple
import jax.lax
import chex

# =============================================================================
# TASK 1: Define Constants and Game State
# =============================================================================

from enum import IntEnum

class Action(IntEnum):
    NO_OP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UP_RIGHT = 6
    UP_LEFT = 7
    DOWN_RIGHT = 8
    DOWN_LEFT = 9
    UP_FIRE = 10
    RIGHT_FIRE = 11
    LEFT_FIRE = 12
    DOWN_FIRE = 13
    UP_RIGHT_FIRE = 14
    UP_LEFT_FIRE = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE = 17

class PlayerID(IntEnum):
    NONE = 0
    PLAYER1_INSIDE = 1
    PLAYER1_OUTSIDE = 2
    PLAYER2_INSIDE = 3
    PLAYER2_OUTSIDE = 4

class OffensivePlay(IntEnum):
    NONE = 0
    PICK_AND_ROLL_LEFT = 1
    PICK_AND_ROLL_RIGHT = 2
    GIVE_AND_GO_LEFT = 3
    GIVE_AND_GO_RIGHT = 4
    PICK_LEFT = 5
    PICK_RIGHT = 6
    MR_INSIDE_SHOOTS = 7
    MR_OUTSIDE_SHOOTS = 8

class DefensivePlay(IntEnum):
    NONE = 0
    LANE_DEFENSE = 1
    TIGHT_DEFENSE_RIGHT = 2
    PASS_DEFENSE_RIGHT = 3
    PICK_DEFENSE_RIGHT = 4
    REBOUND_POSITION_DEFENSE = 5
    PICK_DEFENSE_LEFT = 6
    PASS_DEFENSE_LEFT = 7
    TIGHT_DEFENSE_LEFT = 8

class GameMode(IntEnum):
    PLAY_SELECTION = 0
    IN_PLAY = 1

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
    JUMP_STRENGTH: int = 5 #adjustable if necessary and more of a placeholder value 
    PLAYER_MAX_SPEED: int = 6 #adjustable if necessary and more of a placeholder value
    PLAYER_Y_MIN: int = 50
    PLAYER_Y_MAX: int = 100
    PLAYER_X_MIN: int  = 20
    PLAYER_X_MAX: int = 180
    PLAYER_ROLES: Tuple[int,int] = (0,1) #0 = Offence, 1 = Defence (might be doable with booleans as well)
    BASKET_POSITION: Tuple[int,int] = (100,130)
    BASKET_BUFFER: int = 3 #this should translate to [BASKET_POSITION[0]-buffer:BASKET_POSITION[0]+buffer] being the valid goal area width-wise
    GRAVITY: int = 1 # Downward acceleration due to gravity
    AREA_3_POINT: Tuple[int,int] = (0,0) #We need a way to determine whether a player is in a 3-point area (might be easier to define the two-point area and the rest
                                # will be considered 3-point by process of elimination)


@dataclass
class PlayerState:
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    z: chex.Array
    vel_z: chex.Array
    role: chex.Array # can be 0 for defense, 1 for offense

@dataclass
class BallState:
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    holder: chex.Array # who has the ball (using PlayerID)

@dataclass
class DunkGameState:
    player1_inside: PlayerState
    player1_outside: PlayerState
    player2_inside: PlayerState
    player2_outside: PlayerState
    ball: BallState

    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    acceleration_counter: chex.Array

    # New fields for game logic:
    game_mode: chex.Array           # Current mode (PLAY_SELECTION or IN_PLAY)
    offensive_play: chex.Array      # The selected offensive play
    defensive_play: chex.Array      # The selected defensive play
    play_step: chex.Array           # Tracks progress within a play (e.g., 1st, 2nd, 3rd button press)


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
        self.constants = DunkConstants()

    # =========================================================================
    # TASK 2: Implement Init and Reset
    # =========================================================================
    
    def _init_state(self) -> DunkGameState:
        """
        Creates the very first state of the game.
        Use values from self.constants.
        """
        return DunkGameState(
            player1_inside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            player1_outside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            player2_inside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            player2_outside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            ball=BallState(x=0, y=0, vel_x=0, vel_y=0, holder=PlayerID.NONE),
            player_score=0,
            enemy_score=0,
            step_counter=0,
            acceleration_counter=0,
            game_mode=GameMode.PLAY_SELECTION,
            offensive_play=OffensivePlay.NONE,
            defensive_play=DefensivePlay.NONE,
            play_step=0,
        )

    def _reset_state(self, state: DunkGameState) -> DunkGameState:
        """
        Resets the state after a point is scored or for a new round.
        """
        return DunkGameState(
            player1_inside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            player1_outside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            player2_inside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            player2_outside=PlayerState(x=0, y=0, vel_x=0, vel_y=0, z=0, vel_z=0, role=0),
            ball=BallState(x=0, y=0, vel_x=0, vel_y=0, holder=PlayerID.NONE),
            player_score=state.player_score,
            enemy_score=state.enemy_score,
            game_mode=state.game_mode,
            offensive_play=state.offensive_play,
            defensive_play=state.defensive_play,
            play_step=0,
        )

    def _get_player_action_effects(self, action: int, player_z: chex.Array, constants: DunkConstants) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Determines the velocity for 8-way movement and the impulse for Z-axis jumps.
        """
        # --- X/Y Movement on the ground plane ---
        is_moving_left = (action == Action.LEFT) | (action == Action.UP_LEFT) | (action == Action.DOWN_LEFT) | \
                         (action == Action.LEFT_FIRE) | (action == Action.UP_LEFT_FIRE) | (action == Action.DOWN_LEFT_FIRE)
        is_moving_right = (action == Action.RIGHT) | (action == Action.UP_RIGHT) | (action == Action.DOWN_RIGHT) | \
                          (action == Action.RIGHT_FIRE) | (action == Action.UP_RIGHT_FIRE) | (action == Action.DOWN_RIGHT_FIRE)

        vel_x = jnp.array(0, dtype=jnp.int32)
        vel_x = jax.lax.select(is_moving_left, -constants.PLAYER_MAX_SPEED, vel_x)
        vel_x = jax.lax.select(is_moving_right, constants.PLAYER_MAX_SPEED, vel_x)

        is_moving_up = (action == Action.UP) | (action == Action.UP_LEFT) | (action == Action.UP_RIGHT) | \
                       (action == Action.UP_FIRE) | (action == Action.UP_LEFT_FIRE) | (action == Action.UP_RIGHT_FIRE)
        is_moving_down = (action == Action.DOWN) | (action == Action.DOWN_LEFT) | (action == Action.DOWN_RIGHT) | \
                         (action == Action.DOWN_FIRE) | (action == Action.DOWN_LEFT_FIRE) | (action == Action.DOWN_RIGHT_FIRE)

        vel_y = jnp.array(0, dtype=jnp.int32)
        vel_y = jax.lax.select(is_moving_up, -constants.PLAYER_MAX_SPEED, vel_y)
        vel_y = jax.lax.select(is_moving_down, constants.PLAYER_MAX_SPEED, vel_y)

        # --- Z-Axis Jump Impulse ---
        # A jump can only be initiated if the player is on the ground (z=0) and presses FIRE
        can_jump = (player_z == 0) & (action == Action.FIRE)
        vel_z = jax.lax.select(can_jump, constants.JUMP_STRENGTH, jnp.array(0, dtype=jnp.int32))

        return vel_x, vel_y, vel_z

    def _update_player_physics(self, player: PlayerState, constants: DunkConstants) -> PlayerState:
        """
        Applies physics for both 2D plane movement and Z-axis jumping.
        """
        # --- Z-Axis Physics (Jumping) ---
        # Update Z position based on current Z velocity
        new_z = player.z + player.vel_z

        # Apply gravity for the *next* frame's velocity
        new_vel_z = player.vel_z - constants.GRAVITY

        # Ground collision and state reset
        has_landed = new_z <= 0
        new_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_z)
        new_vel_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_vel_z)

        # --- X/Y Plane Physics (8-way movement) ---
        # Update position
        new_x = player.x + player.vel_x
        new_y = player.y + player.vel_y

        # Screen boundary collision
        new_x = jax.lax.clamp(constants.PLAYER_X_MIN, new_x, constants.PLAYER_X_MAX)
        new_y = jax.lax.clamp(constants.PLAYER_Y_MIN, new_y, constants.PLAYER_Y_MAX)

        return player.replace(x=new_x, y=new_y, z=new_z, vel_z=new_vel_z)

    def _update_player(self, player: PlayerState, action: int, constants: DunkConstants) -> PlayerState:
        """
        Takes a player state and an action, and returns the updated player state
        after applying physics and action effects.
        """
        # Get desired velocity for 8-way movement and any jump impulse
        vel_x, vel_y, vel_z = self._get_player_action_effects(action, player.z, constants)

        # Set the player's velocity based on the action
        updated_player = player.replace(vel_x=vel_x, vel_y=vel_y, vel_z=vel_z)

        # Apply physics (movement, gravity, collisions) to the player
        updated_player = self._update_player_physics(updated_player, constants)

        return updated_player

    # =========================================================================
    # TASK 3: Implement the Step Function
    # =========================================================================
    def step(self, state: DunkGameState, action: int) -> DunkGameState:
        """
        Takes an action in the game and returns the new game state.
        """
        # For now, we assume the user action controls player1_inside. The other players do nothing.
        
        updated_p1_inside = self._update_player(state.player1_inside, action, self.constants)
        updated_p1_outside = self._update_player(state.player1_outside, Action.NO_OP, self.constants)
        updated_p2_inside = self._update_player(state.player2_inside, Action.NO_OP, self.constants)
        updated_p2_outside = self._update_player(state.player2_outside, Action.NO_OP, self.constants)

        # A note on rendering: The player's final visual Y position should be calculated
        # by the renderer as: visual_y = player.y - player.z
        
        return state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )