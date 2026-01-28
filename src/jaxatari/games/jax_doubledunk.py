from jax import numpy as jnp
import jax
from typing import Tuple, NamedTuple
import jax.lax
import jax.debug
import jax.random as random
import chex
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from functools import partial
import os
from enum import IntEnum
from jaxatari import spaces

class PlayerID(IntEnum):
    NONE = 0
    PLAYER1_INSIDE = 1
    PLAYER1_OUTSIDE = 2
    PLAYER2_INSIDE = 3
    PLAYER2_OUTSIDE = 4

class GameMode(IntEnum):
    PLAY_SELECTION = 0
    IN_PLAY = 1
    TRAVEL_PENALTY = 2
    OUT_OF_BOUNDS_PENALTY = 3
    CLEARANCE_PENALTY = 4

class OffensiveAction(IntEnum):
    PASS = 0
    JUMPSHOOT = 1
    MOVE_TO_POST = 2

class DefensiveStrategy(IntEnum):
    LANE_DEFENSE = 0
    TIGHT_DEFENSE = 1
    PASS_DEFENSE = 2
    PICK_DEFENSE = 3
    REBOUND_DEFENSE = 4

# Strategies based on Manual
# Note: "Jump" and "Shoot" are often combined or sequential. 
# We use JUMPSHOOT to represent the phase where shooting is the goal.
PICK_AND_ROLL = jnp.array([OffensiveAction.MOVE_TO_POST, OffensiveAction.PASS, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])
GIVE_AND_GO = jnp.array([OffensiveAction.MOVE_TO_POST, OffensiveAction.PASS, OffensiveAction.PASS, OffensiveAction.JUMPSHOOT])
PICK_PLAY = jnp.array([OffensiveAction.MOVE_TO_POST, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])
MR_INSIDE_SHOOTS = jnp.array([OffensiveAction.PASS, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])
MR_OUTSIDE_SHOOTS = jnp.array([OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])

@chex.dataclass(frozen=True)
class DunkConstants:
    """Holds all static values for the game like screen dimensions, player speeds, colors, etc."""
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210
    BALL_SIZE: Tuple[int, int] = (3,3)
    JUMP_STRENGTH: int = 5
    PLAYER_MAX_SPEED: int = 2
    PLAYER_Y_MIN: int = 57
    PLAYER_Y_MAX: int = 160
    PLAYER_X_MIN: int  = 3
    PLAYER_X_MAX: int = 142
    PLAYER_WIDTH: int = 10                         
    PLAYER_HEIGHT: int = 30
    PLAYER_BARRIER: int = 10  
    BASKET_POSITION: Tuple[int,int] = (80,35)
    GRAVITY: int = 1
    AREA_3_POINT: Tuple[int,int,int] = (25, 135, 90) # (x_min, x_max, y_arc_connect) - needs a proper function to check if a point is in the 3-point area
    MAX_SCORE: int = 24
    DUNK_RADIUS: int = 30
    INSIDE_RADIUS: int = 50
    BLOCK_RADIUS: int = 14
    INSIDE_PLAYER_INSIDE_SHOT = 2
    OUTSIDE_PLAYER_OUTSIDE_SHOT = 2

@chex.dataclass(frozen=True)
class PlayerState:
    id: chex.Array # ID of the Player (see PlayerID) Practically a constant and is primarily used to check if the player is holding a ball for later purposes.
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    z: chex.Array
    vel_z: chex.Array
    role: chex.Array # can be 0 for defense, 1 for offense
    animation_frame: chex.Array
    animation_direction: chex.Array
    is_out_of_bounds: chex.Array # Is the character out of bounds
    jumped_with_ball: chex.Array # Did the character jump while having the ball, jumped_with_ball=False
    triggered_travel: chex.Array # Check if Player broke the "travel rule": jumped and landed while having the ball, triggered_travel=False
    clearance_needed: chex.Array # Does the player need to clear the ball?
    triggered_clearance: chex.Array # Did the player trigger the clearance penalty?

@chex.dataclass(frozen=True)
class BallState:
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    holder: chex.Array # who has the ball (using PlayerID)
    target_x: chex.Array
    target_y: chex.Array
    landing_y: chex.Array
    is_goal: chex.Array # boolean
    shooter: chex.Array
    receiver: chex.Array
    shooter_pos_x: chex.Array
    shooter_pos_y: chex.Array
    missed_shot: chex.Array # boolean, true if ball hit rim and missed

@chex.dataclass(frozen=True)
class GameScores:
    player: chex.Array
    enemy: chex.Array

@chex.dataclass(frozen=True)
class GameTimers:
    offense_cooldown: chex.Array
    enemy_reaction: chex.Array
    travel: chex.Array
    out_of_bounds: chex.Array
    clearance: chex.Array

@chex.dataclass(frozen=True)
class GameStrategy:
    offense_pattern: chex.Array
    defense_pattern: chex.Array
    offense_step: chex.Array
    last_enemy_actions: chex.Array
    play_direction: chex.Array # -1: Left, 1: Right, 0: Center

@chex.dataclass(frozen=True)
class DunkGameState:
    player1_inside: PlayerState
    player1_outside: PlayerState
    player2_inside: PlayerState
    player2_outside: PlayerState
    ball: BallState
    scores: GameScores
    timers: GameTimers
    strategy: GameStrategy
    step_counter: chex.Array
    acceleration_counter: chex.Array
    game_mode: chex.Array      
    controlled_player_id: chex.Array
    key: chex.PRNGKey

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

# Define action sets
_MOVE_LEFT_ACTIONS = {Action.LEFT, Action.UPLEFT, Action.DOWNLEFT}
_MOVE_RIGHT_ACTIONS = {Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT}
_MOVE_UP_ACTIONS = {Action.UP, Action.UPLEFT, Action.UPRIGHT}
_MOVE_DOWN_ACTIONS = {Action.DOWN, Action.DOWNLEFT, Action.DOWNRIGHT}
_JUMP_ACTIONS = {Action.FIRE} 
_PASS_ACTIONS = {Action.FIRE} 
_SHOOT_ACTIONS = {Action.FIRE}
STEAL_ACTIONS = {Action.FIRE}


class DoubleDunk(JaxEnvironment[DunkGameState, DunkObservation, DunkInfo, DunkConstants]):
    
    def __init__(self):
        """Initialize the game environment."""
        self.constants = DunkConstants()
        self.renderer = DunkRenderer(self.constants)

    def reset(self, key) -> Tuple[DunkObservation, DunkGameState]:
        """Resets the environment to the initial state."""
        state = self._init_state(key)
        obs = self._get_observation(state)
        return obs, state

    def _get_observation(self, state: DunkGameState) -> DunkObservation:
        """Converts the environment state to an observation."""
        player = EntityPosition(
            x=jnp.array(state.player1_inside.x, dtype=jnp.int32),
            y=jnp.array(state.player1_inside.y, dtype=jnp.int32),
            width=jnp.array(self.constants.PLAYER_WIDTH, dtype=jnp.int32),  
            height=jnp.array(self.constants.PLAYER_HEIGHT, dtype=jnp.int32), 
        )
        enemy = EntityPosition(
            x=jnp.array(state.player2_inside.x, dtype=jnp.int32),
            y=jnp.array(state.player2_inside.y, dtype=jnp.int32),
            width=jnp.array(self.constants.PLAYER_WIDTH, dtype=jnp.int32),  
            height=jnp.array(self.constants.PLAYER_HEIGHT, dtype=jnp.int32), 
        )
        ball = EntityPosition(
            x=jnp.array(state.ball.x, dtype=jnp.int32),
            y=jnp.array(state.ball.y, dtype=jnp.int32),
            width=jnp.array(self.constants.BALL_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.constants.BALL_SIZE[1], dtype=jnp.int32),
        )
        return DunkObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            score_player=state.scores.player.astype(jnp.int32),
            score_enemy=state.scores.enemy.astype(jnp.int32),
        )

    def action_space(self):
        """Returns the action space of the environment."""
        return spaces.Discrete(18)
    
    def observation_space(self):
        """Returns the observation space of the environment."""
        field=spaces.Dict({
                "x": spaces.Box(low=0, high=200, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=240, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=200, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=240, shape=(), dtype=jnp.int32),
            })
            
        return spaces.Dict({
            "player": field,
            "enemy": field,
            "ball": field,
            "score_player": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "score_enemy": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
        })
    
    def image_space(self) -> spaces.Box:
        """Returns the image space of the environment."""
        return spaces.Box(low=0, high=255, shape=(self.constants.WINDOW_HEIGHT, self.constants.WINDOW_WIDTH, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs: DunkObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""
        return jnp.concatenate([
            obs.player.x.reshape(-1),
            obs.player.y.reshape(-1),
            obs.player.width.reshape(-1),
            obs.player.height.reshape(-1),
            obs.enemy.x.reshape(-1),
            obs.enemy.y.reshape(-1),
            obs.enemy.width.reshape(-1),
            obs.enemy.height.reshape(-1),
            obs.ball.x.reshape(-1),
            obs.ball.y.reshape(-1),
            obs.ball.width.reshape(-1),
            obs.ball.height.reshape(-1),
            obs.score_player.reshape(-1),
            obs.score_enemy.reshape(-1),
        ])

    def _init_state(self, key, holder=PlayerID.PLAYER1_OUTSIDE) -> DunkGameState:
        """Creates the very first state of the game."""
        
        is_p2_holder = jnp.logical_or((holder == PlayerID.PLAYER2_INSIDE), (holder == PlayerID.PLAYER2_OUTSIDE))
        
        p1_out_y = jax.lax.select(is_p2_holder, jnp.array(145, dtype=jnp.int32), jnp.array(155, dtype=jnp.int32))
        p2_out_y = jax.lax.select(is_p2_holder, jnp.array(155, dtype=jnp.int32), jnp.array(145, dtype=jnp.int32))

        return DunkGameState(
            player1_inside=PlayerState(id=1, x=100, y=70, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            player1_outside=PlayerState(id=2, x=75, y=p1_out_y, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            player2_inside=PlayerState(id=3, x=50, y=60, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            player2_outside=PlayerState(id=4, x=75, y=p2_out_y, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            ball=BallState(x=75.0, y=150.0, vel_x=0.0, vel_y=0.0, holder=holder, target_x=0.0, target_y=0.0, landing_y=0.0, is_goal=False, shooter=PlayerID.NONE, receiver=PlayerID.NONE, shooter_pos_x=0, shooter_pos_y=0, missed_shot=False),
            scores=GameScores(
                player=jnp.array(0, dtype=jnp.int32),
                enemy=jnp.array(0, dtype=jnp.int32)
            ),
            timers=GameTimers(
                offense_cooldown=0,
                enemy_reaction=0,
                travel=0,
                out_of_bounds=0,
                clearance=0
            ),
            strategy=GameStrategy(
                offense_pattern=GIVE_AND_GO,
                defense_pattern=DefensiveStrategy.LANE_DEFENSE,
                offense_step=0,
                last_enemy_actions=jnp.array([Action.NOOP, Action.NOOP]),
                play_direction=0
            ),
            step_counter=0,
            acceleration_counter=0,
            game_mode=GameMode.PLAY_SELECTION,
            controlled_player_id = jax.lax.select(
                jnp.logical_or((holder == PlayerID.PLAYER1_INSIDE), (holder == PlayerID.PLAYER2_INSIDE)),
                PlayerID.PLAYER1_INSIDE,
                PlayerID.PLAYER1_OUTSIDE
            ),
            key=key,
        )

    def _get_player_xy_action_effects(self, action: int, constants: DunkConstants) -> Tuple[chex.Array, chex.Array]:
        """Determines the velocity for 8-way movement."""
        action_jnp = jnp.asarray(action)

        is_moving_left = jnp.any(action_jnp == jnp.asarray(list(_MOVE_LEFT_ACTIONS)))
        is_moving_right = jnp.any(action_jnp == jnp.asarray(list(_MOVE_RIGHT_ACTIONS)))

        vel_x = jnp.array(0, dtype=jnp.int32)
        vel_x = jax.lax.select(is_moving_left, -constants.PLAYER_MAX_SPEED, vel_x)
        vel_x = jax.lax.select(is_moving_right, constants.PLAYER_MAX_SPEED, vel_x)

        is_moving_up = jnp.any(action_jnp == jnp.asarray(list(_MOVE_UP_ACTIONS)))
        is_moving_down = jnp.any(action_jnp == jnp.asarray(list(_MOVE_DOWN_ACTIONS)))

        vel_y = jnp.array(0, dtype=jnp.int32)
        vel_y = jax.lax.select(is_moving_up, -constants.PLAYER_MAX_SPEED, vel_y)
        vel_y = jax.lax.select(is_moving_down, constants.PLAYER_MAX_SPEED, vel_y)

        return vel_x, vel_y

    def _update_player_xy(self, player: PlayerState, action: int, constants: DunkConstants) -> PlayerState:
        """Updates the player's XY position based on an action."""
        vel_x, vel_y = self._get_player_xy_action_effects(action, constants)
        updated_x = player.x + vel_x
        updated_y = player.y + vel_y
        new_x = jax.lax.clamp(constants.PLAYER_X_MIN, updated_x, constants.PLAYER_X_MAX)
        new_y = jax.lax.clamp(constants.PLAYER_Y_MIN, updated_y, constants.PLAYER_Y_MAX)
        touched_bound = jnp.logical_or(jnp.logical_or((updated_x <= constants.PLAYER_X_MIN), (updated_x >= constants.PLAYER_X_MAX)), (updated_y <= constants.PLAYER_Y_MIN))

        # Clearance Check: Check if player is "inside". If not, they have cleared the ball.
        # Inside Zone definition based on scoring logic:
        # 1. Rectangular Zone: x=[25, 135], y <= 90
        in_rect_zone = jnp.logical_and(jnp.logical_and((new_x >= 25), (new_x <= 135)), (new_y <= 90))
        # 2. Elliptical Zone: Center(80, 80), Rx=55, Ry=45
        dx = new_x - 80.0
        dy = new_y - 80.0
        ellipse_val = (dx**2 / (55.0**2)) + (dy**2 / (45.0**2))
        in_ellipse_zone = jnp.logical_and((ellipse_val <= 1.0), (new_y >= 80))
        
        is_inside = jnp.logical_or(in_rect_zone, in_ellipse_zone)
        is_outside = jnp.logical_not(is_inside)
        
        new_clearance_needed = jax.lax.select(is_outside, False, player.clearance_needed)

        return player.replace(x=new_x, y=new_y, vel_x=vel_x, vel_y=vel_y, is_out_of_bounds=touched_bound, clearance_needed=new_clearance_needed)

    def _update_players_xy(self, state: DunkGameState, actions: Tuple[int, ...]) -> DunkGameState:
        """Updates the XY positions for all players."""
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        updated_players = jax.vmap(self._update_player_xy, in_axes=(0, 0, None))(players, actions_stacked, self.constants)
        
        updated_p1_inside, updated_p1_outside, updated_p2_inside, updated_p2_outside = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

        # Check if any of the players reach out of bounds while holding the ball
        ball_holder_id = state.ball.holder
        p1_outside_out_of_bounds = jnp.logical_and(updated_p1_outside.is_out_of_bounds, (updated_p1_outside.id == ball_holder_id))
        p1_inside_out_of_bounds = jnp.logical_and(updated_p1_inside.is_out_of_bounds, (updated_p1_inside.id == ball_holder_id))
        p1_out_of_bounds = jnp.logical_or(p1_inside_out_of_bounds, p1_outside_out_of_bounds) # if Player 1 triggered out of bounds

        # --- Updated Game State ---
        updated_state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )

        # Instead of resetting immediately, we switch to OUT_OF_BOUNDS_PENALTY mode
        # Freeze for ~1 second (60 frames)
        penalty_state = updated_state.replace(
            game_mode=GameMode.OUT_OF_BOUNDS_PENALTY,
            timers=updated_state.timers.replace(out_of_bounds=60),
        )

        new_state = jax.lax.cond(p1_out_of_bounds, lambda x: penalty_state, lambda x: updated_state, None)

        return new_state

    def _resolve_penalty_and_reset(self, state: DunkGameState, p1_at_fault: bool) -> DunkGameState:
        """Shared logic to reset the game after a penalty and switch possession."""
        key, reset_key = random.split(state.key)
        new_ball_holder = jax.lax.select(p1_at_fault, PlayerID.PLAYER2_OUTSIDE, PlayerID.PLAYER1_OUTSIDE)
        
        # Create a fresh state but preserve scores and step counter
        fresh_state = self._init_state(reset_key, holder=new_ball_holder)
        
        return fresh_state.replace(
            scores=state.scores,
            step_counter=state.step_counter,
        )

    def _handle_out_of_bounds_penalty(self, state: DunkGameState) -> DunkGameState:
        """Handles the out of bounds penalty freeze."""
        
        # Decrement timer
        new_timer = state.timers.out_of_bounds - 1
        timer_expired = new_timer <= 0

        # We need to know who held the ball to know who triggered it.
        # But the ball holder hasn't changed yet in 'state'.
        ball_holder_id = state.ball.holder
        
        p1_inside_out_of_bounds = jnp.logical_and(state.player1_inside.is_out_of_bounds, (state.player1_inside.id == ball_holder_id))
        p1_outside_out_of_bounds = jnp.logical_and(state.player1_outside.is_out_of_bounds, (state.player1_outside.id == ball_holder_id))
        p1_at_fault = jnp.logical_or(p1_inside_out_of_bounds, p1_outside_out_of_bounds)

        return jax.lax.cond(
            timer_expired,
            lambda s: self._resolve_penalty_and_reset(s, p1_at_fault),
            lambda s: s.replace(timers=s.timers.replace(out_of_bounds=new_timer)),
            state
        )

    def _update_player_z(self, player: PlayerState, constants: DunkConstants, ball_hold_id: int) -> PlayerState:
        """Applies Z-axis physics (jumping and gravity) to a player."""
        has_ball = (ball_hold_id == player.id) #check if the player has the ball
        new_z = player.z + player.vel_z
        new_vel_z = player.vel_z - constants.GRAVITY
        has_landed = new_z <= 0 #check if the player is back on the ground
        jump_start_with_ball = jnp.logical_and((player.z == 0), has_ball) #check if the player starts a jump while having the ball
        new_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_z)
        new_vel_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_vel_z)
        has_triggered_travel = jnp.logical_and(has_landed, player.jumped_with_ball) # True if the player lands while having the ball at the start of the jump
        # update PlayerState value jumped_with_ball
        updated_jumped_with_ball = jax.lax.select(jump_start_with_ball, jnp.logical_not(has_landed),
                                    jax.lax.select(jnp.logical_not(has_landed), jnp.logical_and(has_ball, player.jumped_with_ball), False))
        # update PlayerState with new height and whether or not they started/finished the jump with the ball
        return player.replace(z=new_z, vel_z=new_vel_z, jumped_with_ball=updated_jumped_with_ball, triggered_travel=has_triggered_travel)

    def _update_players_z(self, state: DunkGameState) -> DunkGameState:
        """Applies Z-axis physics for all players."""
        ball_holder_id = state.ball.holder
        
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        
        updated_players = jax.vmap(self._update_player_z, in_axes=(0, None, None))(players, self.constants, ball_holder_id)
        
        updated_p1_inside, updated_p1_outside, updated_p2_inside, updated_p2_outside = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

        # check if any players triggered the travel rule
        travel_triggered = jnp.logical_or(updated_p1_outside.triggered_travel, updated_p1_inside.triggered_travel)


        # --- Updated Game State ---
        updated_state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )

        # Instead of resetting immediately, we switch to TRAVEL_PENALTY mode
        # Freeze for ~1 second (60 frames)
        penalty_state = updated_state.replace(
            game_mode=GameMode.TRAVEL_PENALTY,
            timers=updated_state.timers.replace(travel=60)
        )

        new_state = jax.lax.cond(travel_triggered, lambda x: penalty_state, lambda x: updated_state, None)

        return new_state

    def _update_players_animations(self, state: DunkGameState) -> DunkGameState:
        """Updates animations for all players."""
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])
        has_ball_arr = (state.ball.holder == player_ids)
        
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        
        updated_players = jax.vmap(self._update_player_animation, in_axes=(0, 0))(players, has_ball_arr)
        
        updated_p1_inside, updated_p1_outside, updated_p2_inside, updated_p2_outside = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

        return state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )

    def _handle_miss(self, state: DunkGameState) -> DunkGameState:
        """Handles a missed shot by making the ball fall to the ground."""
        ball = state.ball
        # Set ball to fall straight down from its current position
        new_ball = ball.replace(
            vel_x=0.0, 
            vel_y=2.0,  # Positive y is downwards
            landing_y=float(self.constants.PLAYER_Y_MIN+30), # Ground level
            receiver=PlayerID.NONE, # Reset receiver
            missed_shot=True
        )
        return state.replace(ball=new_ball)

    def _handle_player_actions(self, state: DunkGameState, action: int, key: chex.PRNGKey) -> Tuple[Tuple[int, ...], chex.PRNGKey, chex.Array, chex.Array]:
        """Determines the action for each player based on control state and AI."""
        
        def get_move_to_target(current_x, current_y, target_x, target_y, threshold=10):
            dx = target_x - current_x
            dy = target_y - current_y
            
            # 1. Determine direction booleans enforcing the deadzone
            # The agent only moves if the distance is STRICTLY greater than threshold
            want_right = dx > threshold
            want_left  = dx < -threshold
            want_down  = dy > threshold
            want_up    = dy < -threshold
            
            # 2. Start with NOOP (this covers the case where all flags are False)
            action = Action.NOOP
            
            # 3. Apply Cardinals (Single axis movement)
            # We update 'action' sequentially.
            action = jax.lax.select(want_up, Action.UP, action)
            action = jax.lax.select(want_down, Action.DOWN, action)
            action = jax.lax.select(want_left, Action.LEFT, action)
            action = jax.lax.select(want_right, Action.RIGHT, action)
            
            # 4. Apply Diagonals (Dual axis movement)
            # These override the cardinals if BOTH flags are true.
            action = jax.lax.select(jnp.logical_and(want_up, want_left), Action.UPLEFT, action)
            action = jax.lax.select(jnp.logical_and(want_up, want_right), Action.UPRIGHT, action)
            action = jax.lax.select(jnp.logical_and(want_down, want_left), Action.DOWNLEFT, action)
            action = jax.lax.select(jnp.logical_and(want_down, want_right), Action.DOWNRIGHT, action)
            
            return action

        # --- Human Control ---
        is_p1_inside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_INSIDE)
        is_p1_outside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_OUTSIDE)
        p1_inside_action = jax.lax.select(is_p1_inside_controlled, action, Action.NOOP)
        p1_outside_action = jax.lax.select(is_p1_outside_controlled, action, Action.NOOP)

        # --- AI Logic Setup ---
        key, p2_action_key, teammate_action_key = random.split(key, 3)
        # We need 4 keys for enemies: 2 for probability checks and 2 for random move selection
        p2_inside_prob_key, p2_outside_prob_key, p2_inside_move_key, p2_outside_move_key = random.split(p2_action_key, 4)
        movement_actions = jnp.array([
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.UPLEFT, Action.UPRIGHT, Action.DOWNLEFT, Action.DOWNRIGHT
        ])

        # --- Strategy & Possession ---
        p1_has_ball = jnp.logical_or((state.ball.holder == PlayerID.PLAYER1_INSIDE), (state.ball.holder == PlayerID.PLAYER1_OUTSIDE))
        p2_has_ball = jnp.logical_or((state.ball.holder == PlayerID.PLAYER2_INSIDE), (state.ball.holder == PlayerID.PLAYER2_OUTSIDE))
        
        defensive_strat = state.strategy.defense_pattern
        basket_x, basket_y = self.constants.BASKET_POSITION

        # --- Defensive Logic (Target Calculation) ---
        
        # Helper to select target based on strategy
        def select_def_target(strat, t_lane, t_tight, t_pass, t_pick, t_reb):
            return jax.lax.select(
                strat == DefensiveStrategy.LANE_DEFENSE, t_lane,
                jax.lax.select(
                    strat == DefensiveStrategy.TIGHT_DEFENSE, t_tight,
                    jax.lax.select(
                        strat == DefensiveStrategy.PASS_DEFENSE, t_pass,
                        jax.lax.select(
                            strat == DefensiveStrategy.PICK_DEFENSE, t_pick,
                            t_reb # REBOUND
                        )
                    )
                )
            )

        # --- P1 Defensive Targets (if P1 is defending) ---
        # Opponents: P2 Inside, P2 Outside
        
        # 1. TIGHT (Man-to-man)
        p1_in_t_tight_x, p1_in_t_tight_y = state.player2_inside.x, state.player2_inside.y
        p1_out_t_tight_x, p1_out_t_tight_y = state.player2_outside.x, state.player2_outside.y
        
        # 2. LANE (Inside helps middle)
        p1_in_t_lane_x = (state.player2_inside.x + state.player2_outside.x) // 2
        p1_in_t_lane_y = (state.player2_inside.y + state.player2_outside.y) // 2
        p1_out_t_lane_x, p1_out_t_lane_y = p1_out_t_tight_x, p1_out_t_tight_y
        
        # 3. PASS (Guard the passing lane - midpoint)
        mid_p2_x = (state.player2_inside.x + state.player2_outside.x) // 2
        mid_p2_y = (state.player2_inside.y + state.player2_outside.y) // 2
        p1_in_t_pass_x, p1_in_t_pass_y = mid_p2_x, mid_p2_y
        p1_out_t_pass_x, p1_out_t_pass_y = mid_p2_x, mid_p2_y
        
        # 4. PICK (Switch on close proximity)
        dist_p2 = jnp.sqrt((state.player2_inside.x - state.player2_outside.x)**2 + (state.player2_inside.y - state.player2_outside.y)**2)
        switch_p1 = dist_p2 < 15
        p1_in_t_pick_x = jax.lax.select(switch_p1, state.player2_outside.x, state.player2_inside.x)
        p1_in_t_pick_y = jax.lax.select(switch_p1, state.player2_outside.y, state.player2_inside.y)
        p1_out_t_pick_x = jax.lax.select(switch_p1, state.player2_inside.x, state.player2_outside.x)
        p1_out_t_pick_y = jax.lax.select(switch_p1, state.player2_inside.y, state.player2_outside.y)
        
        # 5. REBOUND (Inside guards basket)
        p1_in_t_reb_x, p1_in_t_reb_y = basket_x, basket_y + 10
        p1_out_t_reb_x, p1_out_t_reb_y = p1_out_t_tight_x, p1_out_t_tight_y

        p1_in_def_x = select_def_target(defensive_strat, p1_in_t_lane_x, p1_in_t_tight_x, p1_in_t_pass_x, p1_in_t_pick_x, p1_in_t_reb_x)
        p1_in_def_y = select_def_target(defensive_strat, p1_in_t_lane_y, p1_in_t_tight_y, p1_in_t_pass_y, p1_in_t_pick_y, p1_in_t_reb_y)
        p1_out_def_x = select_def_target(defensive_strat, p1_out_t_lane_x, p1_out_t_tight_x, p1_out_t_pass_x, p1_out_t_pick_x, p1_out_t_reb_x)
        p1_out_def_y = select_def_target(defensive_strat, p1_out_t_lane_y, p1_out_t_tight_y, p1_out_t_pass_y, p1_out_t_pick_y, p1_out_t_reb_y)
        
        p1_in_def_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, p1_in_def_x, p1_in_def_y)
        p1_out_def_action = get_move_to_target(state.player1_outside.x, state.player1_outside.y, p1_out_def_x, p1_out_def_y)

        # --- P2 Defensive Targets (if P2 is defending) ---
        # Opponents: P1 Inside, P1 Outside

        # 1. TIGHT
        p2_in_t_tight_x, p2_in_t_tight_y = state.player1_inside.x, state.player1_inside.y
        p2_out_t_tight_x, p2_out_t_tight_y = state.player1_outside.x, state.player1_outside.y

        # 2. LANE
        p2_in_t_lane_x = (state.player1_inside.x + state.player1_outside.x) // 2
        p2_in_t_lane_y = (state.player1_inside.y + state.player1_outside.y) // 2
        p2_out_t_lane_x, p2_out_t_lane_y = p2_out_t_tight_x, p2_out_t_tight_y

        # 3. PASS
        mid_p1_x = (state.player1_inside.x + state.player1_outside.x) // 2
        mid_p1_y = (state.player1_inside.y + state.player1_outside.y) // 2
        p2_in_t_pass_x, p2_in_t_pass_y = mid_p1_x, mid_p1_y
        p2_out_t_pass_x, p2_out_t_pass_y = mid_p1_x, mid_p1_y

        # 4. PICK
        dist_p1 = jnp.sqrt((state.player1_inside.x - state.player1_outside.x)**2 + (state.player1_inside.y - state.player1_outside.y)**2)
        switch_p2 = dist_p1 < 15
        p2_in_t_pick_x = jax.lax.select(switch_p2, state.player1_outside.x, state.player1_inside.x)
        p2_in_t_pick_y = jax.lax.select(switch_p2, state.player1_outside.y, state.player1_inside.y)
        p2_out_t_pick_x = jax.lax.select(switch_p2, state.player1_inside.x, state.player1_outside.x)
        p2_out_t_pick_y = jax.lax.select(switch_p2, state.player1_inside.y, state.player1_outside.y)

        # 5. REBOUND
        p2_in_t_reb_x, p2_in_t_reb_y = basket_x, basket_y + 10
        p2_out_t_reb_x, p2_out_t_reb_y = p2_out_t_tight_x, p2_out_t_tight_y

        p2_in_def_x = select_def_target(defensive_strat, p2_in_t_lane_x, p2_in_t_tight_x, p2_in_t_pass_x, p2_in_t_pick_x, p2_in_t_reb_x)
        p2_in_def_y = select_def_target(defensive_strat, p2_in_t_lane_y, p2_in_t_tight_y, p2_in_t_pass_y, p2_in_t_pick_y, p2_in_t_reb_y)
        p2_out_def_x = select_def_target(defensive_strat, p2_out_t_lane_x, p2_out_t_tight_x, p2_out_t_pass_x, p2_out_t_pick_x, p2_out_t_reb_x)
        p2_out_def_y = select_def_target(defensive_strat, p2_out_t_lane_y, p2_out_t_tight_y, p2_out_t_pass_y, p2_out_t_pick_y, p2_out_t_reb_y)

        p2_in_def_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, p2_in_def_x, p2_in_def_y)
        p2_out_def_action = get_move_to_target(state.player2_outside.x, state.player2_outside.y, p2_out_def_x, p2_out_def_y)

        # --- Teammate AI (Offensive/Random vs Defensive) ---
        is_p1_inside_teammate_ai = jnp.logical_not(is_p1_inside_controlled)
        is_p1_outside_teammate_ai = jnp.logical_not(is_p1_outside_controlled)

        # Move to Post Logic (Active if PREVIOUS step was MOVE_TO_POST)
        # The manual says "First press: Move to post". This implies the behavior lasts UNTIL the next press.
        prev_step = state.strategy.offense_step - 1
        prev_action_was_move = jnp.logical_and((prev_step >= 0), (state.strategy.offense_pattern[prev_step] == OffensiveAction.MOVE_TO_POST))
        
        # Determine Post Position based on chosen direction
        # Left Post: 40, Right Post: 120, Center: 80 (y=60)
        play_dir = state.strategy.play_direction
        post_x = jax.lax.select(play_dir == -1, 40, jax.lax.select(play_dir == 1, 120, 80))
        post_y = 60
        
        p1_inside_dist_to_post = jnp.sqrt((state.player1_inside.x - post_x)**2 + (state.player1_inside.y - post_y)**2)
        p1_at_post = p1_inside_dist_to_post < 5
        
        # If we are in the "Move to Post" phase, override random movement until we arrive
        should_move_to_post = jnp.logical_and(prev_action_was_move, jnp.logical_not(p1_at_post))
        p1_move_to_post_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, post_x, post_y)

        # Random/Heuristic Offensive Moves (Existing logic)
        random_teammate_move_idx = random.randint(teammate_action_key, shape=(), minval=0, maxval=8)
        random_teammate_move_action = movement_actions[random_teammate_move_idx]
        rand_teammate = random.uniform(teammate_action_key)
        
        p1_off_action = jax.lax.select(rand_teammate < 0.5, Action.NOOP, random_teammate_move_action)

        # Chase Logic
        ball_is_free = (state.ball.holder == PlayerID.NONE)
        p1_in_chase = get_move_to_target(state.player1_inside.x, state.player1_inside.y, state.ball.x, state.ball.y, 2)
        p1_out_chase = get_move_to_target(state.player1_outside.x, state.player1_outside.y, state.ball.x, state.ball.y, 2)
        p2_in_chase = get_move_to_target(state.player2_inside.x, state.player2_inside.y, state.ball.x, state.ball.y, 2)
        p2_out_chase = get_move_to_target(state.player2_outside.x, state.player2_outside.y, state.ball.x, state.ball.y, 2)

        # Constraint check for P1 Inside Offense (Teammate)
        p1_dist_to_basket_x = jnp.abs(state.player1_inside.x - basket_x)
        p1_dist_to_basket_y = jnp.abs(state.player1_inside.y - basket_y)
        p1_is_far = jnp.logical_or((p1_dist_to_basket_x > 20), (p1_dist_to_basket_y > 80)) # 40x80 area
        p1_return_to_basket_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, basket_x, basket_y)
        
        p1_inside_action = jax.lax.select(
            is_p1_inside_teammate_ai,
            jax.lax.select(
                ball_is_free,
                p1_in_chase,
                jax.lax.select(
                    p2_has_ball, 
                    p1_in_def_action, 
                    jax.lax.select(
                        should_move_to_post,
                        p1_move_to_post_action,
                        jax.lax.select(p1_is_far, p1_return_to_basket_action, p1_off_action)
                    )
                )
            ),
            p1_inside_action
        )
        p1_outside_action = jax.lax.select(
            is_p1_outside_teammate_ai,
            jax.lax.select(
                ball_is_free,
                p1_out_chase,
                jax.lax.select(p2_has_ball, p1_out_def_action, p1_off_action)
            ),
            p1_outside_action
        )

        # --- P2 AI (Offensive/Heuristic vs Defensive) ---
        
        # P2 Inside Offense
        p2_inside_has_ball = (state.ball.holder == PlayerID.PLAYER2_INSIDE)
        rand_inside = random.uniform(p2_inside_prob_key)
        random_inside_move_idx = random.randint(p2_inside_move_key, shape=(), minval=0, maxval=8)
        random_inside_move_action = movement_actions[random_inside_move_idx]
        action_if_ball_inside = jax.lax.select(rand_inside < 0.2, Action.FIRE, random_inside_move_action)
        
        # Constraint check for P2 Inside Offense
        p2_dist_to_basket_x = jnp.abs(state.player2_inside.x - basket_x)
        p2_dist_to_basket_y = jnp.abs(state.player2_inside.y - basket_y)
        p2_is_far = jnp.logical_or((p2_dist_to_basket_x > 20), (p2_dist_to_basket_y > 40)) # 40x80 area
        p2_return_to_basket_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, basket_x, basket_y)

        p2_in_off_action = jax.lax.select(
            p2_inside_has_ball, 
            action_if_ball_inside, 
            jax.lax.select(
                p2_is_far, 
                p2_return_to_basket_action, 
                jax.lax.select(rand_inside < 0.5, Action.NOOP, random_inside_move_action)
            )
        )

        # P2 Outside Offense
        p2_outside_has_ball = (state.ball.holder == PlayerID.PLAYER2_OUTSIDE)
        rand_outside = random.uniform(p2_outside_prob_key)
        random_outside_move_idx = random.randint(p2_outside_move_key, shape=(), minval=0, maxval=8)
        random_outside_move_action = movement_actions[random_outside_move_idx]
        action_if_ball_outside = jax.lax.select(rand_outside < 0.2, Action.FIRE, random_outside_move_action)
        p2_out_off_action = jax.lax.select(p2_outside_has_ball, action_if_ball_outside, jax.lax.select(rand_outside < 0.5, Action.NOOP, random_outside_move_action))

        # P2 Clearance Override (Smart Target)
        def get_smart_clearance_target(px, py):
            px_f = px.astype(jnp.float32)
            py_f = py.astype(jnp.float32)
            
            # 1. Distances to sides
            dist_left = px_f - 25.0
            dist_right = 135.0 - px_f
            
            # 2. Distance to ellipse bottom
            dx = px_f - 80.0
            term = 1.0 - (dx / 55.0)**2
            valid_term = jnp.maximum(0.0, term)
            boundary_y = 80.0 + 45.0 * jnp.sqrt(valid_term)
            dist_down = boundary_y - py_f
            
            # 3. Determine closest escape
            go_left = jnp.logical_and((dist_left < dist_right), (dist_left < dist_down))
            go_right = jnp.logical_and((dist_right <= dist_left), (dist_right < dist_down))
            # go_down is implicit else
            
            tx = jax.lax.select(go_left, 20.0, jax.lax.select(go_right, 140.0, px_f))
            ty = jax.lax.select(jnp.logical_or(go_left, go_right), py_f, boundary_y + 10.0)
            
            return tx.astype(jnp.int32), ty.astype(jnp.int32)


        p2_in_tx, p2_in_ty = get_smart_clearance_target(state.player2_inside.x, state.player2_inside.y)
        p2_in_clearance_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, p2_in_tx, p2_in_ty)
        p2_in_final_off = jax.lax.select(state.player2_inside.clearance_needed, p2_in_clearance_action, p2_in_off_action)

        p2_out_tx, p2_out_ty = get_smart_clearance_target(state.player2_outside.x, state.player2_outside.y)
        p2_out_clearance_action = get_move_to_target(state.player2_outside.x, state.player2_outside.y, p2_out_tx, p2_out_ty)
        p2_out_final_off = jax.lax.select(state.player2_outside.clearance_needed, p2_out_clearance_action, p2_out_off_action)

        # Final P2 Actions (Defensive if P1 has ball, Chase if free, Offensive otherwise)
        p2_inside_action = jax.lax.select(
            ball_is_free,
            p2_in_chase,
            jax.lax.select(p1_has_ball, p2_in_def_action, p2_in_final_off)
        )
        
        p2_outside_action = jax.lax.select(
            ball_is_free,
            p2_out_chase,
            jax.lax.select(p1_has_ball, p2_out_def_action, p2_out_final_off)
        )

        # --- Defensive Jump Logic (CPU Defenders) ---
        # When the attacker (P1) is in the air with the ball during a shot, the CPU defenders should jump
        is_shoot_step = (state.strategy.offense_pattern[state.strategy.offense_step] == OffensiveAction.JUMPSHOOT)
        
        # Check if P1 Inside or P1 Outside is in the air with the ball
        p1_inside_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE)
        p1_outside_has_ball = (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        p1_inside_in_air = state.player1_inside.z > 0
        p1_outside_in_air = state.player1_outside.z > 0
        
        p1_inside_shooting = jnp.logical_and(jnp.logical_and(p1_inside_has_ball, p1_inside_in_air), is_shoot_step)
        p1_outside_shooting = jnp.logical_and(jnp.logical_and(p1_outside_has_ball, p1_outside_in_air), is_shoot_step)
        p1_is_shooting = jnp.logical_or(p1_inside_shooting, p1_outside_shooting)
        
        # Determine if P2 is defending and close to the shooter
        p2_in_close_to_p1_in = jnp.sqrt((state.player2_inside.x - state.player1_inside.x)**2 + (state.player2_inside.y - state.player1_inside.y)**2) < 40
        p2_in_close_to_p1_out = jnp.sqrt((state.player2_inside.x - state.player1_outside.x)**2 + (state.player2_inside.y - state.player1_outside.y)**2) < 40
        p2_out_close_to_p1_in = jnp.sqrt((state.player2_outside.x - state.player1_inside.x)**2 + (state.player2_outside.y - state.player1_inside.y)**2) < 40
        p2_out_close_to_p1_out = jnp.sqrt((state.player2_outside.x - state.player1_outside.x)**2 + (state.player2_outside.y - state.player1_outside.y)**2) < 40
        
        # P2 Inside defensive jump: jumps if P1 Inside is shooting and P2 Inside is close
        p2_in_defensive_jump = jnp.logical_and(p1_inside_shooting, p2_in_close_to_p1_in)
        
        # P2 Outside defensive jump: jumps if P1 Outside is shooting and P2 Outside is close
        p2_out_defensive_jump = jnp.logical_and(p1_outside_shooting, p2_out_close_to_p1_out)
        
        # P2 Inside can also jump if P1 Outside is shooting and P2 Inside is close (interior defense)
        p2_in_defensive_jump = jnp.logical_or(p2_in_defensive_jump, jnp.logical_and(p1_outside_shooting, p2_in_close_to_p1_out))
        
        # P2 Outside can also jump if P1 Inside is shooting and P2 Outside is close (perimeter help)
        p2_out_defensive_jump = jnp.logical_or(p2_out_defensive_jump, jnp.logical_and(p1_inside_shooting, p2_out_close_to_p1_in))
        
        # Add FIRE (Jump) action if defensive jump is needed
        p2_inside_def_action = jax.lax.select(
            jnp.logical_and(p2_in_defensive_jump, p1_has_ball),
            Action.FIRE,
            p2_inside_action
        )
        
        p2_outside_def_action = jax.lax.select(
            jnp.logical_and(p2_out_defensive_jump, p1_has_ball),
            Action.FIRE,
            p2_outside_action
        )

        # --- Enemy Reaction Time Logic ---
        use_last_action = state.timers.enemy_reaction > 0
        
        final_p2_inside_action = jax.lax.select(use_last_action, state.strategy.last_enemy_actions[0], p2_inside_def_action)
        final_p2_outside_action = jax.lax.select(use_last_action, state.strategy.last_enemy_actions[1], p2_outside_def_action)
        
        new_timer = jax.lax.select(use_last_action, state.timers.enemy_reaction - 1, 6) # Reset to 6 frames (~0.1s)
        new_last_actions = jax.lax.select(use_last_action, state.strategy.last_enemy_actions, jnp.array([p2_inside_def_action, p2_outside_def_action]))

        actions = (p1_inside_action, p1_outside_action, final_p2_inside_action, final_p2_outside_action)
        return actions, key, new_timer, new_last_actions
    
    def _update_player_animation(self, player: PlayerState, has_ball: bool) -> PlayerState:
        """Updates the animation frame for a single player."""
        anim_frame = player.animation_frame
        anim_dir = player.animation_direction

        # Calculate next frame
        new_dir = jax.lax.cond(anim_frame >= 9, lambda: -1, lambda: anim_dir)
        new_dir = jax.lax.cond(anim_frame <= 0, lambda: 1, lambda: new_dir)
        new_frame = anim_frame + new_dir

        # Update if player has the ball, otherwise reset
        final_frame = jax.lax.select(has_ball, new_frame, 0)
        final_dir = jax.lax.select(has_ball, new_dir, 1)

        return player.replace(animation_frame=final_frame, animation_direction=final_dir)

    def _handle_passing(self, state: DunkGameState, actions: Tuple[int, ...]) -> Tuple[BallState, chex.Array, chex.Array]:
        """Handles the logic for passing the ball."""
        ball_state = state.ball
        is_pass_step = (state.strategy.offense_pattern[state.strategy.offense_step] == OffensiveAction.PASS)
        
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        def check_passing(pid, p_action):
            is_passing = jnp.logical_and(jnp.logical_and(jnp.logical_and((state.timers.offense_cooldown == 0), is_pass_step), (ball_state.holder == pid)), jnp.any(jnp.asarray(p_action) == jnp.asarray(list(_PASS_ACTIONS))))
            return is_passing

        passing_flags = jax.vmap(check_passing)(player_ids, actions_stacked)
        is_passing = jnp.any(passing_flags)
        passer_idx = jnp.argmax(passing_flags)
        
        # Receivers: 1->0, 0->1, 3->2, 2->3 (indices in player_ids)
        receiver_indices = jnp.array([1, 0, 3, 2])
        receiver_idx = receiver_indices[passer_idx]
        receiver = player_ids[receiver_idx]

        passer_x = players.x[passer_idx]
        passer_y = players.y[passer_idx]
        receiver_x = players.x[receiver_idx]
        receiver_y = players.y[receiver_idx]

        passer_pos = jnp.array([passer_x, passer_y], dtype=jnp.float32)
        receiver_pos = jnp.array([receiver_x, receiver_y], dtype=jnp.float32)
        direction = receiver_pos - passer_pos
        norm = jnp.sqrt(jnp.sum(direction**2))
        safe_norm = jnp.where(norm == 0, 1.0, norm)
        ball_speed = 8.0
        vel = (direction / safe_norm) * ball_speed

        new_ball_state = jax.lax.cond(
            is_passing,
            lambda b: b.replace(x=passer_x.astype(jnp.float32), y=passer_y.astype(jnp.float32), vel_x=vel[0], vel_y=vel[1], holder=PlayerID.NONE, receiver=receiver),
            lambda b: b,
            ball_state
        )

        step_increment = jax.lax.select(is_passing, 1, 0)
        return new_ball_state, step_increment, is_passing

    def _handle_shooting(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[BallState, chex.PRNGKey, chex.Array, chex.Array]:
        """Handles the logic for shooting the ball."""
        ball_state = state.ball
        is_shoot_step = (state.strategy.offense_pattern[state.strategy.offense_step] == OffensiveAction.JUMPSHOOT)
        
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        def check_shooting(pid, p_z, p_action):
            is_shooting = jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and((state.timers.offense_cooldown == 0), is_shoot_step), (p_z != 0)), (ball_state.holder == pid)), jnp.any(jnp.asarray(p_action) == jnp.asarray(list(_SHOOT_ACTIONS))))
            return is_shooting

        shooting_flags = jax.vmap(check_shooting)(player_ids, players.z, actions_stacked)
        is_shooting = jnp.any(shooting_flags)
        shooter_idx = jnp.argmax(shooting_flags)
        shooter = player_ids[shooter_idx]
        shooter_x = players.x[shooter_idx]
        shooter_y = players.y[shooter_idx]
        shooter_z = players.z[shooter_idx]

        is_inside_shooting = jnp.logical_or((shooter == PlayerID.PLAYER1_INSIDE), (shooter == PlayerID.PLAYER2_INSIDE))
        is_outside_shooting = jnp.logical_or((shooter == PlayerID.PLAYER1_OUTSIDE), (shooter == PlayerID.PLAYER2_OUTSIDE))

        key, offset_key_x = random.split(key)

        shooter_pos = jnp.array([shooter_x, shooter_y], dtype=jnp.float32)
        basket_pos = jnp.array([self.constants.BASKET_POSITION[0], self.constants.BASKET_POSITION[1]], dtype=jnp.float32)
        dist_to_basket = jnp.sqrt(jnp.sum((shooter_pos - basket_pos) ** 2))
        
        # --- Dunk Logic ---
        is_dunk = jnp.logical_and((dist_to_basket < self.constants.DUNK_RADIUS), (shooter_z > 0))
        
        is_inside = dist_to_basket < self.constants.INSIDE_RADIUS
        shot_bonus = jax.lax.select(jnp.logical_and(is_inside, is_inside_shooting), 2, jax.lax.select(jnp.logical_and(jnp.logical_not(is_inside), is_outside_shooting), 2, -2))

        offset_x = random.uniform(offset_key_x, shape=(), minval=-10 + shot_bonus, maxval=10 - shot_bonus)
        
        # If it's a dunk, no offset and guaranteed goal
        final_offset_x = jax.lax.select(is_dunk, 0.0, offset_x)
        is_goal = jax.lax.select(is_dunk, True, jnp.logical_and((final_offset_x >= -5), (final_offset_x <= 5)))

        target_pos = basket_pos + jnp.array([final_offset_x, 0.0])

        shoot_direction = target_pos - shooter_pos
        shoot_norm = jnp.sqrt(jnp.sum(shoot_direction**2))
        shoot_safe_norm = jnp.where(shoot_norm == 0, 1.0, shoot_norm)
        shoot_speed = 8.0
        shoot_vel = (shoot_direction / shoot_safe_norm) * shoot_speed

        # Interception/Blocking Logic
        # A shot can only be blocked if the defending player is in the air (z > 0)
        # This requires the defender to jump when the attacker jumps to shoot
        def check_blocking():
            # opponents (if shooter is P1, opponents are P2; if shooter is P2, opponents are P1)
            is_shooter_p1 = jnp.logical_or((shooter == PlayerID.PLAYER1_INSIDE), (shooter == PlayerID.PLAYER1_OUTSIDE))
            
            opp_indices = jax.lax.select(is_shooter_p1, jnp.array([2, 3]), jnp.array([0, 1]))
            opp_xs = players.x[opp_indices]
            opp_ys = players.y[opp_indices]
            opp_zs = players.z[opp_indices]
            opp_ids = player_ids[opp_indices]

            dists = jnp.sqrt((opp_xs - shooter_x)**2 + (opp_ys - shooter_y)**2)
            # Defender must be in the air (opp_zs > 0) AND within BLOCK_RADIUS to intercept
            can_blocks = jnp.logical_and((opp_zs > 0), (dists < self.constants.BLOCK_RADIUS))
            
            blocked_by = jax.lax.select(can_blocks[0], opp_ids[0], jax.lax.select(can_blocks[1], opp_ids[1], PlayerID.NONE))
            return blocked_by

        blocked_by = jax.lax.cond(is_shooting, check_blocking, lambda: PlayerID.NONE)

        def make_shot(b):
            b = b.replace(x=shooter_x.astype(jnp.float32), y=shooter_y.astype(jnp.float32), vel_x=shoot_vel[0], vel_y=shoot_vel[1], holder=PlayerID.NONE, target_x=target_pos[0], target_y=target_pos[1], is_goal=is_goal, shooter=shooter, receiver=PlayerID.NONE, shooter_pos_x=shooter_x.astype(jnp.int32), shooter_pos_y=shooter_y.astype(jnp.int32))
            b = jax.lax.cond(blocked_by != PlayerID.NONE, lambda bb: bb.replace(holder=blocked_by, vel_x=0.0, vel_y=0.0, is_goal=False, shooter=PlayerID.NONE), lambda bb: bb, b)
            return b

        new_ball_state = jax.lax.cond(
            is_shooting,
            make_shot,
            lambda b: b,
            ball_state
        )
        step_increment = jax.lax.select(is_shooting, 1, 0)
        return new_ball_state, key, step_increment, is_shooting

    def _handle_stealing(self, state: DunkGameState, actions: Tuple[int, ...]) -> Tuple[BallState, chex.Array, chex.Array]:
        """Handles the logic for stealing the ball. Returns ball state, success flags, and attempt flags."""
        ball_state = state.ball
        
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])

        # Get ball holder Z position to check for dribbling
        holder_idx = jnp.clip(ball_state.holder - 1, 0, 3)
        holder_z = players.z[holder_idx]
        is_dribbling = (holder_z == 0)

        def check_steal(player, action, pid):
            is_trying_to_steal = jnp.any(jnp.asarray(action) == jnp.asarray(list(STEAL_ACTIONS)))
            dist_sq = (player.x - ball_state.x)**2 + (player.y - ball_state.y)**2
            is_close_to_ball = dist_sq < 25.0
            
            # Can only steal if opponent has ball
            # Team 1: IDs 1, 2. Team 2: IDs 3, 4.
            is_p1_team = (pid <= 2)
            holder_is_p1_team = jnp.logical_or((ball_state.holder == PlayerID.PLAYER1_INSIDE), (ball_state.holder == PlayerID.PLAYER1_OUTSIDE))
            holder_is_p2_team = jnp.logical_or((ball_state.holder == PlayerID.PLAYER2_INSIDE), (ball_state.holder == PlayerID.PLAYER2_OUTSIDE))
            
            opponent_has_ball = jnp.logical_or(jnp.logical_and(is_p1_team, holder_is_p2_team), jnp.logical_and(jnp.logical_not(is_p1_team), holder_is_p1_team))
            
            # Steal Mode is active if opponent has ball AND is dribbling (grounded)
            steal_mode_active = jnp.logical_and(opponent_has_ball, is_dribbling)
            
            # Attempt is valid if mode is active AND button is pressed
            is_attempt = jnp.logical_and(steal_mode_active, is_trying_to_steal)
            
            # Success requires attempt AND close distance
            is_success = jnp.logical_and(is_attempt, is_close_to_ball)

            return is_success, is_attempt

        steal_flags, attempt_flags = jax.vmap(check_steal)(players, actions_stacked, player_ids)
        
        is_stealing = jnp.any(steal_flags)
        stealer_idx = jnp.argmax(steal_flags)
        stealer_id = player_ids[stealer_idx]

        new_ball_state = jax.lax.cond(
            is_stealing,
            lambda b: b.replace(holder=stealer_id, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            ball_state
        )
        return new_ball_state, steal_flags, attempt_flags

    def _handle_jump(self, state: DunkGameState, player: PlayerState, action: int, constants: DunkConstants) -> chex.Array:
        """Calculates the vertical impulse for a jump."""
        is_jump_step = jnp.logical_and((state.strategy.offense_pattern[state.strategy.offense_step] == OffensiveAction.JUMPSHOOT), (player.z == 0))
        can_jump = jnp.logical_and(jnp.logical_and((state.timers.offense_cooldown == 0), is_jump_step), jnp.any(jnp.asarray(action) == jnp.asarray(list(_JUMP_ACTIONS))))
        vel_z = jax.lax.select(can_jump, constants.JUMP_STRENGTH, jnp.array(0, dtype=jnp.int32))
        new_vel_z = jax.lax.select(vel_z > 0, vel_z, player.vel_z)
        did_jump = jax.lax.select(can_jump, 1, 0)

        # Clearance Penalty Check
        has_ball = (state.ball.holder == player.id)
        triggered_clearance = jnp.logical_and(jnp.logical_and(can_jump, has_ball), player.clearance_needed)

        return player.replace(vel_z=new_vel_z, triggered_clearance=triggered_clearance), did_jump

    def _handle_offense_actions(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[DunkGameState, chex.PRNGKey, chex.Array]:
        """Handles offensive actions: passing, shooting, and move commands."""
        
        # Check current action type
        current_step_action = state.strategy.offense_pattern[state.strategy.offense_step]
        is_move_step = (current_step_action == OffensiveAction.MOVE_TO_POST)
        
        # Check trigger for move step (Any Fire press by holder's team)
        ball_holder = state.ball.holder
        holder_idx = jnp.clip(ball_holder - 1, 0, 3) # 0..3
        
        # Map actions to simple array
        actions_arr = jnp.array(actions)
        holder_action = jax.lax.select(
            (ball_holder != PlayerID.NONE),
            actions_arr[holder_idx],
            Action.NOOP
        )
        
        is_fire_pressed = jnp.any(jnp.asarray(holder_action) == jnp.asarray(list(_PASS_ACTIONS)))
        
        move_inc = jax.lax.select(
            jnp.logical_and(jnp.logical_and(is_move_step, is_fire_pressed), (state.timers.offense_cooldown == 0)),
            1,
            0
        )
        
        # Passing
        ball_state_after_pass, pass_inc, did_pass = self._handle_passing(state, actions)
        state = state.replace(ball=ball_state_after_pass)
        state = state.replace(timers=state.timers.replace(offense_cooldown=jax.lax.select(jnp.logical_or(did_pass, (move_inc > 0)), 6, state.timers.offense_cooldown)))

        # Shooting
        ball_state_after_shot, key, shot_inc, did_shoot = self._handle_shooting(state, actions, key)
        state = state.replace(ball=ball_state_after_shot)
        state = state.replace(timers=state.timers.replace(offense_cooldown=jax.lax.select(did_shoot, 6, state.timers.offense_cooldown)))

        step_increment = pass_inc + shot_inc + move_inc
        return state, key, step_increment

    def _handle_interactions(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[DunkGameState, chex.PRNGKey]:
        """Handles all player interactions: jump, pass, shoot, steal."""
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        # 1. Handle Stealing FIRST
        ball_state_after_steal, steal_flags, attempt_flags = self._handle_stealing(state, actions)
        state = state.replace(ball=ball_state_after_steal)
        
        # Set a cooldown if a steal was successful to prevent immediate jump/shoot
        is_stolen = jnp.any(steal_flags)
        state = state.replace(timers=state.timers.replace(offense_cooldown=jax.lax.select(is_stolen, 6, state.timers.offense_cooldown)))

        # Mask actions for players who ATTEMPTED to steal so they don't Jump/Pass/Shoot
        # If attempt_flag is true, it means Fire was pressed in Steal Mode.
        # We should mask it to NOOP to prevent it from triggering a Jump.
        def mask_action(act, did_attempt):
             return jax.lax.select(did_attempt, Action.NOOP, act)
        masked_actions_stacked = jax.vmap(mask_action)(actions_stacked, attempt_flags)
        
        # Convert masked stack back to tuple for other functions if needed, 
        # but _handle_jump uses stack, _handle_offense uses tuple.
        masked_actions = tuple([masked_actions_stacked[i] for i in range(4)])

        # 2. Handle Jump (using masked actions)
        updated_players, jumped_flags = jax.vmap(self._handle_jump, in_axes=(None, 0, 0, None))(state, players, masked_actions_stacked, self.constants)
        
        updated_p1_inside, updated_p1_outside, updated_p2_inside, updated_p2_outside = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

        did_jump = jnp.max(jumped_flags)

        state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )
        state = state.replace(timers=state.timers.replace(offense_cooldown=jax.lax.select(did_jump > 0, 6, state.timers.offense_cooldown)))

        # 3. Handle Offense (Pass/Shoot) using masked actions
        state, key, offense_increment = self._handle_offense_actions(state, masked_actions, key)
        # _handle_defense_actions removed as it is handled in step 1

        # 4. Update offensive_strategy_step
        new_offensive_strategy_step = jnp.minimum(state.strategy.offense_step + offense_increment, len(state.strategy.offense_pattern)-1)
        
        # 5. Print if changed
        # We use jax.lax.cond to ensure we only print when an action actually occurred
        jax.lax.cond(
            offense_increment > 0,
            lambda x: jax.debug.print("Play Step: {}", x),
            lambda x: None,
            new_offensive_strategy_step
        )

        return state.replace(strategy=state.strategy.replace(offense_step=new_offensive_strategy_step)), key

    def _update_ball(self, state: DunkGameState) -> DunkGameState:
        """Handles ball movement, goals, misses, catches, and possession changes."""
        ball_in_flight = (state.ball.holder == PlayerID.NONE)
        dist_to_target = jnp.sqrt((state.ball.x - state.ball.target_x)**2 + (state.ball.y - state.ball.target_y)**2)
        is_already_falling = jnp.logical_and((state.ball.vel_x == 0), jnp.isclose(state.ball.vel_y, 2.0))
        reached_target = jnp.logical_and(jnp.logical_and(ball_in_flight, (dist_to_target < 5.0)), jnp.logical_not(is_already_falling))
        is_goal_scored = jnp.logical_and(reached_target, state.ball.is_goal)

        def on_goal(s):
            key, reset_key = random.split(s.key)
            is_p1_scorer = jnp.logical_or((s.ball.shooter == PlayerID.PLAYER1_INSIDE), (s.ball.shooter == PlayerID.PLAYER1_OUTSIDE))

            # --- 2 vs 3 Point Logic ---
            # Based on the background generation script geometry:
            # 1. Rectangular Zone: x=[25, 135], y <= 90
            # 2. Elliptical Zone: Center(80, 80), Rx=55, Ry=45 (approx due to perspective)
            
            sx = s.ball.shooter_pos_x
            sy = s.ball.shooter_pos_y

            # Check 1: Rectangular Key Area
            # The script draws vertical lines at x=25 and x=135 down to y=90
            in_rect_zone = jnp.logical_and(jnp.logical_and((sx >= 25), (sx <= 135)), (sy <= 90))

            # Check 2: Elliptical Top of Key
            # Equation: ((x-h)/rx)^2 + ((y-k)/ry)^2 <= 1
            # Center (h,k) = (80, 80)
            dx = sx - 80.0
            dy = sy - 80.0
            
            # We use float division for the ellipse calculation
            ellipse_val = (dx**2 / (55.0**2)) + (dy**2 / (45.0**2))
            
            # We only care about the ellipse part that extends below the center (y >= 80)
            in_ellipse_zone = jnp.logical_and((ellipse_val <= 1.0), (sy >= 80))

            # A shot is 2 points if it is in EITHER zone. Otherwise, it's a 3-pointer.
            is_2_point = jnp.logical_or(in_rect_zone, in_ellipse_zone)
            points = jax.lax.select(is_2_point, 2, 3)

            # --- Score Updates ---
            new_player_score = s.scores.player + points * is_p1_scorer
            new_enemy_score = s.scores.enemy + points * (1 - is_p1_scorer)
            
            # --- Reset Game State ---
            # Give ball to the team that got scored on
            new_ball_holder = jax.lax.select(is_p1_scorer, PlayerID.PLAYER2_OUTSIDE, PlayerID.PLAYER1_OUTSIDE)

            # We recreate the initial state but preserve scores and step counter
            new_state = self._init_state(reset_key, holder=new_ball_holder).replace(
                scores=s.scores.replace(player=new_player_score, enemy=new_enemy_score), 
                step_counter=s.step_counter
            )
            
            return new_state

        def continue_play(s):
            # Handle miss
            # Check if ball went past the basket line (Y < 10) while moving up
            overshot = jnp.logical_and((s.ball.y < self.constants.BASKET_POSITION[1]), (s.ball.vel_y < 0))
            is_miss = jnp.logical_or(jnp.logical_and(reached_target, jnp.logical_not(s.ball.is_goal)), overshot)
            
            s = jax.lax.cond(is_miss, self._handle_miss, lambda s_: s_, s)
            b_state = s.ball

            players_stacked = jax.tree_util.tree_map(lambda *args: jnp.stack(args), s.player1_inside, s.player1_outside, s.player2_inside, s.player2_outside)
            player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])

            # --- New logic for passing ---
            is_passing = b_state.receiver != PlayerID.NONE

            def update_pass_trajectory(b):
                receiver_x = players_stacked.x[b.receiver - 1]
                receiver_y = players_stacked.y[b.receiver - 1]

                passer_pos = jnp.array([b.x, b.y], dtype=jnp.float32)
                receiver_pos = jnp.array([receiver_x, receiver_y], dtype=jnp.float32)
                direction = receiver_pos - passer_pos
                norm = jnp.sqrt(jnp.sum(direction**2))
                safe_norm = jnp.where(norm == 0, 1.0, norm)
                ball_speed = 8.0
                vel = (direction / safe_norm) * ball_speed
                return b.replace(vel_x=vel[0], vel_y=vel[1])

            b_state = jax.lax.cond(is_passing, update_pass_trajectory, lambda b: b, b_state)

            # Update ball flight physics
            is_falling = jnp.logical_and((b_state.vel_x == 0), jnp.isclose(b_state.vel_y, 2.0))
            new_ball_x = b_state.x + b_state.vel_x
            new_ball_y = b_state.y + b_state.vel_y
            has_landed = jnp.logical_and(is_falling, (new_ball_y >= b_state.landing_y))
            final_y = jax.lax.select(has_landed, b_state.landing_y, new_ball_y)
            final_vel_y = jax.lax.select(has_landed, 0.0, b_state.vel_y)
            b_state = jax.lax.cond(
                (b_state.holder == PlayerID.NONE),
                lambda b: b.replace(x=new_ball_x, y=final_y, vel_y=final_vel_y),
                lambda b: b,
                b_state
            )

            # Handle interceptions and catches
            catch_radius_sq = 25.0 
            ball_in_flight_after_physics = (b_state.holder == PlayerID.NONE)

            def check_catch_flag(px, py):
                dist_sq = (b_state.x - px)**2 + (b_state.y - py)**2
                return jnp.logical_and(ball_in_flight_after_physics, (dist_sq < catch_radius_sq))

            catch_flags = jax.vmap(check_catch_flag)(players_stacked.x, players_stacked.y)
            any_caught = jnp.any(catch_flags)
            # To match the original sequential loop where the LAST player wins:
            # We use argmax on reversed flags
            reversed_catcher_idx = jnp.argmax(catch_flags[::-1])
            catcher_idx = 3 - reversed_catcher_idx

            def handle_catch(curr_state, curr_ball):
                pid = player_ids[catcher_idx]
                
                # Check for clearance logic (Rebound)
                shooter_id = curr_ball.shooter
                is_rebound = jnp.logical_and((shooter_id != PlayerID.NONE), curr_ball.missed_shot)
                
                shooter_team_p1 = jnp.logical_or((shooter_id == PlayerID.PLAYER1_INSIDE), (shooter_id == PlayerID.PLAYER1_OUTSIDE))
                catcher_team_p1 = jnp.logical_or((pid == PlayerID.PLAYER1_INSIDE), (pid == PlayerID.PLAYER1_OUTSIDE))
                is_defensive_rebound = jnp.logical_and(is_rebound, (shooter_team_p1 != catcher_team_p1))
                
                # Update ball
                new_ball = curr_ball.replace(holder=pid, vel_x=0.0, vel_y=0.0, receiver=PlayerID.NONE, shooter=PlayerID.NONE, missed_shot=False)
                
                # Update player state using vmap
                players_stacked_catch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), curr_state.player1_inside, curr_state.player1_outside, curr_state.player2_inside, curr_state.player2_outside)
                
                def update_clearance(player, p_id):
                    should_set = jnp.logical_and((p_id == pid), is_defensive_rebound)
                    return jax.lax.cond(should_set, lambda p: p.replace(clearance_needed=True), lambda p: p, player)

                updated_players = jax.vmap(update_clearance)(players_stacked_catch, player_ids)
                
                u_p1_in, u_p1_out, u_p2_in, u_p2_out = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

                return curr_state.replace(ball=new_ball, player1_inside=u_p1_in, player1_outside=u_p1_out, player2_inside=u_p2_in, player2_outside=u_p2_out)

            # We need to update state, not just ball
            s = jax.lax.cond(any_caught, lambda s_: handle_catch(s_, b_state), lambda s_: s_.replace(ball=b_state), s)
            
            # Refresh ball state from potentially updated s
            b_state = s.ball

            # Update ball position if held
            players_stacked_new = jax.tree_util.tree_map(lambda *args: jnp.stack(args), s.player1_inside, s.player1_outside, s.player2_inside, s.player2_outside)
            is_held = jnp.logical_and((b_state.holder >= PlayerID.PLAYER1_INSIDE), (b_state.holder <= PlayerID.PLAYER2_OUTSIDE))
            def update_held_pos(b):
                idx = b.holder - 1
                return b.replace(x=players_stacked_new.x[idx].astype(jnp.float32), y=players_stacked_new.y[idx].astype(jnp.float32))
            
            b_state = jax.lax.cond(is_held, update_held_pos, lambda b: b, b_state)

            # Update controlled player
            holder = b_state.holder
            should_switch_to_inside = jnp.logical_or((holder == PlayerID.PLAYER1_INSIDE), (holder == PlayerID.PLAYER2_INSIDE))
            should_switch_to_outside = jnp.logical_or((holder == PlayerID.PLAYER1_OUTSIDE), (holder == PlayerID.PLAYER2_OUTSIDE))
            
            new_controlled_player_id = jax.lax.select(
                should_switch_to_inside,
                PlayerID.PLAYER1_INSIDE,
                jax.lax.select(
                    should_switch_to_outside,
                    PlayerID.PLAYER1_OUTSIDE,
                    s.controlled_player_id
                )
            )
            
            return s.replace(ball=b_state, controlled_player_id=new_controlled_player_id)

        final_state = jax.lax.cond(is_goal_scored, on_goal, continue_play, state)
        return final_state

    def _handle_play_selection(self, state: DunkGameState, action: int) -> DunkGameState:
        """Handles the play selection mode."""
        p1_has_ball = jnp.logical_or((state.ball.holder == PlayerID.PLAYER1_INSIDE), (state.ball.holder == PlayerID.PLAYER1_OUTSIDE))
        
        key, strat_key = random.split(state.key)

        # Enemy choices
        random_offensive_idx = random.randint(strat_key, shape=(), minval=0, maxval=5) # 0, 1, 2, 3, 4
        enemy_offensive_strategy = jax.lax.switch(random_offensive_idx, [
            lambda: PICK_AND_ROLL, 
            lambda: GIVE_AND_GO, 
            lambda: MR_OUTSIDE_SHOOTS,
            lambda: PICK_PLAY,
            lambda: MR_INSIDE_SHOOTS
        ])
        
        random_defensive_idx = random.randint(strat_key, shape=(), minval=0, maxval=5) # 0, 1, 2, 3, 4
        enemy_defensive_strategy = random_defensive_idx

        def start_game(s, selected_off, selected_def, direction):
            # When a strategy is selected, we reset the game to its initial state
            # but keep the selected strategy and switch to IN_PLAY mode.
            # We also keep the current key.
            return s.replace(
                strategy=GameStrategy(
                    offense_pattern=selected_off,
                    defense_pattern=selected_def,
                    offense_step=0,
                    last_enemy_actions=jnp.array([Action.NOOP, Action.NOOP]),
                    play_direction=direction
                ),
                game_mode=GameMode.IN_PLAY,
                key=key
            )

        is_up = (action == Action.UPFIRE)
        is_down = (action == Action.DOWNFIRE)
        is_right = (action == Action.RIGHTFIRE)
        is_left = (action == Action.LEFTFIRE)
        is_up_left = (action == Action.UPLEFTFIRE)
        is_up_right = (action == Action.UPRIGHTFIRE)
        is_down_left = (action == Action.DOWNLEFTFIRE)
        is_down_right = (action == Action.DOWNRIGHTFIRE)

        # Case 1: P1 Has Ball (User chooses Offense, Enemy chooses Defense)
        # Mappings aligned with manual:
        # TOP (UP)          = MR_INSIDE_SHOOTS
        # BOTTOM (DOWN)     = MR_OUTSIDE_SHOOTS
        # L/R               = GIVE_AND_GO
        # UPPER L/R         = PICK_AND_ROLL
        # LOWER L/R         = PICK_PLAY
        
        off_p1 = jax.lax.select(
            is_up, MR_INSIDE_SHOOTS, 
            jax.lax.select(
                is_down, MR_OUTSIDE_SHOOTS, 
                jax.lax.select(
                    jnp.logical_or(is_left, is_right), GIVE_AND_GO,
                    jax.lax.select(
                        jnp.logical_or(is_up_left, is_up_right), PICK_AND_ROLL,
                        jax.lax.select(
                            jnp.logical_or(is_down_left, is_down_right), PICK_PLAY,
                            state.strategy.offense_pattern
                        )
                    )
                )
            )
        )
        
        # Determine Play Direction (-1: Left, 1: Right, 0: Center)
        # Left variants: LEFT, UP_LEFT, DOWN_LEFT
        # Right variants: RIGHT, UP_RIGHT, DOWN_RIGHT
        play_dir_p1 = jax.lax.select(
            jnp.logical_or(jnp.logical_or(is_left, is_up_left), is_down_left), -1,
            jax.lax.select(
                jnp.logical_or(jnp.logical_or(is_right, is_up_right), is_down_right), 1,
                0 # UP and DOWN are center
            )
        )
        
        valid_input_p1_off = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(is_up, is_down), is_right), is_left), is_up_left), is_up_right), is_down_left), is_down_right)
        
        state_p1_ball = jax.lax.cond(
            valid_input_p1_off,
            lambda s: start_game(s, off_p1, enemy_defensive_strategy, play_dir_p1),
            lambda s: s,
            state
        )

        # Case 2: P2 Has Ball (Enemy chooses Offense, User chooses Defense)
        # Mapping from Manual:
        # TOP (UP)          = LANE_DEFENSE
        # UPPER L/R         = TIGHT_DEFENSE
        # L/R               = PASS_DEFENSE
        # LOWER L/R         = PICK_DEFENSE
        # BOTTOM (DOWN)     = REBOUND_DEFENSE

        def_p1 = jax.lax.select(
            is_up, DefensiveStrategy.LANE_DEFENSE,
            jax.lax.select(
                jnp.logical_or(is_up_left, is_up_right), DefensiveStrategy.TIGHT_DEFENSE,
                jax.lax.select(
                    jnp.logical_or(is_left, is_right), DefensiveStrategy.PASS_DEFENSE,
                    jax.lax.select(
                        jnp.logical_or(is_down_left, is_down_right), DefensiveStrategy.PICK_DEFENSE,
                        jax.lax.select(
                            is_down, DefensiveStrategy.REBOUND_DEFENSE,
                            state.strategy.defense_pattern
                        )
                    )
                )
            )
        )
        
        # Enemy picks a direction (Randomly)
        enemy_play_dir = random.randint(strat_key, shape=(), minval=-1, maxval=2) # -1, 0, 1 
        
        valid_input_p1_def = valid_input_p1_off

        state_p2_ball = jax.lax.cond(
             valid_input_p1_def,
             lambda s: start_game(s, enemy_offensive_strategy, def_p1, enemy_play_dir),
             lambda s: s,
             state
        )

        return jax.lax.cond(p1_has_ball, lambda: state_p1_ball, lambda: state_p2_ball)

    def _handle_travel_penalty(self, state: DunkGameState) -> DunkGameState:
        """Handles the travel penalty freeze."""
        
        # Decrement timer
        new_timer = state.timers.travel - 1
        timer_expired = new_timer <= 0

        p1_triggered = jnp.logical_or(state.player1_inside.triggered_travel, state.player1_outside.triggered_travel)

        return jax.lax.cond(
            timer_expired,
            lambda s: self._resolve_penalty_and_reset(s, p1_triggered),
            lambda s: s.replace(timers=s.timers.replace(travel=new_timer)),
            state
        )

    def _handle_clearance_penalty(self, state: DunkGameState) -> DunkGameState:
        """Handles the clearance penalty freeze."""
        
        # Decrement timer
        new_timer = state.timers.clearance - 1
        timer_expired = new_timer <= 0

        p1_triggered = jnp.logical_or(state.player1_inside.triggered_clearance, state.player1_outside.triggered_clearance)

        return jax.lax.cond(
            timer_expired,
            lambda s: self._resolve_penalty_and_reset(s, p1_triggered),
            lambda s: s.replace(timers=s.timers.replace(clearance=new_timer)),
            state
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DunkGameState, action: int) -> Tuple[DunkObservation, DunkGameState, float, bool, DunkInfo]:
        """Takes an action in the game and returns the new game state."""

        # Handle play selection mode
        state = jax.lax.cond(
            state.game_mode == GameMode.PLAY_SELECTION,
            lambda s: self._handle_play_selection(s, action),
            lambda s: s,
            state
        )

        # Handle travel penalty mode
        state = jax.lax.cond(
            state.game_mode == GameMode.TRAVEL_PENALTY,
            lambda s: self._handle_travel_penalty(s),
            lambda s: s,
            state
        )

        # Handle out of bounds penalty mode
        state = jax.lax.cond(
            state.game_mode == GameMode.OUT_OF_BOUNDS_PENALTY,
            lambda s: self._handle_out_of_bounds_penalty(s),
            lambda s: s,
            state
        )

        # Handle clearance penalty mode
        state = jax.lax.cond(
            state.game_mode == GameMode.CLEARANCE_PENALTY,
            lambda s: self._handle_clearance_penalty(s),
            lambda s: s,
            state
        )

        def run_game_step(s):
            # Update cooldown (decrement)
            new_cooldown = jnp.maximum(0, s.timers.offense_cooldown - 1)
            s = s.replace(timers=s.timers.replace(offense_cooldown=new_cooldown))

            # 1. Determine actions for all players
            actions, key, new_timer, new_last_actions = self._handle_player_actions(s, action, s.key)
            s = s.replace(
                timers=s.timers.replace(enemy_reaction=new_timer),
                strategy=s.strategy.replace(last_enemy_actions=new_last_actions)
            )

            # 2. Update player XY movement
            s = self._update_players_xy(s, actions)

            # 3. Handle interactions (jump, pass, shoot, steal)
            s, key = self._handle_interactions(s, actions, key)
            
            # Check for Clearance Penalty Trigger (after jump)
            p1_clearance = jnp.logical_or(s.player1_inside.triggered_clearance, s.player1_outside.triggered_clearance)
            p2_clearance = jnp.logical_or(s.player2_inside.triggered_clearance, s.player2_outside.triggered_clearance)
            clearance_triggered = jnp.logical_or(p1_clearance, p2_clearance)
            
            s = jax.lax.cond(
                clearance_triggered,
                lambda s_: s_.replace(
                    game_mode=GameMode.CLEARANCE_PENALTY,
                    timers=s_.timers.replace(clearance=60)
                ),
                lambda s_: s_,
                s
            )

            # 4. Update player Z physics (gravity, etc.)
            s = self._update_players_z(s)

            # 5. Update player animations
            s = self._update_players_animations(s)

            # 6. Process ball flight, goals, misses, and possession changes
            final_s = self._update_ball(s)

            final_s = final_s.replace(step_counter=final_s.step_counter + 1, key=key)
            return final_s

        # Run the game step only if in IN_PLAY mode
        final_state = jax.lax.cond(
            state.game_mode == GameMode.IN_PLAY,
            run_game_step,
            lambda s: s.replace(step_counter=s.step_counter + 1), 
            state
        )

        # 7. Generate outputs
        observation = self._get_observation(final_state)
        reward = self._get_reward(state, final_state)
        done = self._get_done(final_state)
        info = self._get_info(final_state)

        return observation, final_state, reward, done, info

    def _get_reward(self, previous_state: DunkGameState, state: DunkGameState) -> float:
        """Calculates the reward from the environment state."""
        # Placeholder: return 0 reward for now
        return 0.0

    def _get_done(self, state: DunkGameState) -> bool:
        """Determines if the environment state is a terminal state"""
        is_max_score_reached = jnp.logical_or((state.scores.player >= self.constants.MAX_SCORE), (state.scores.enemy >= self.constants.MAX_SCORE))
        return is_max_score_reached

    def _get_info(self, state: DunkGameState, all_rewards: jnp.array = None) -> DunkInfo:
        """Extracts information from the environment state."""
        # Placeholder: return step count
        return DunkInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DunkGameState) -> jnp.ndarray:
        return self.renderer.render(state)

class DunkRenderer(JAXGameRenderer):
    def __init__(self, consts: DunkConstants = None):
        super().__init__()
        self.consts = consts or DunkConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # --- UPDATED ASSET CONFIGURATION ---
        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},

            # Team 1 (Player) - Blue/Black
            {'name': 'player_inside_1', 'type': 'single', 'file': 'player_inside_1.npy'},
            {'name': 'player_inside_guard', 'type': 'single', 'file': 'player_inside_guard.npy'},
            {'name': 'player_inside_jump', 'type': 'single', 'file': 'player_inside_jump.npy'},
            {'name': 'player_outside_1', 'type': 'single', 'file': 'player_outside_1.npy'},
            {'name': 'player_outside_guard', 'type': 'single', 'file': 'player_outside_guard.npy'},
            {'name': 'player_outside_jump', 'type': 'single', 'file': 'player_outside_jump.npy'},

            # Team 2 (Enemy) - Red/White
            {'name': 'enemy_inside_1', 'type': 'single', 'file': 'enemy_inside_1.npy'},
            {'name': 'enemy_inside_guard', 'type': 'single', 'file': 'enemy_inside_guard.npy'},
            {'name': 'enemy_inside_jump', 'type': 'single', 'file': 'enemy_inside_jump.npy'},
            {'name': 'enemy_outside_1', 'type': 'single', 'file': 'enemy_outside_1.npy'},
            {'name': 'enemy_outside_guard', 'type': 'single', 'file': 'enemy_outside_guard.npy'},
            {'name': 'enemy_outside_jump', 'type': 'single', 'file': 'enemy_outside_jump.npy'},

            # Indicators
            {'name': 'player_off', 'type': 'single', 'file': 'player_off.npy'},
            {'name': 'player_def', 'type': 'single', 'file': 'player_def.npy'},
            {'name': 'enemy_off', 'type': 'single', 'file': 'enemy_off.npy'},
            {'name': 'enemy_def', 'type': 'single', 'file': 'enemy_def.npy'},

            {'name': 'player_score', 'type': 'digits', 'pattern': 'player_score_{}.npy', 'files': [f'player_score_{i}.npy' for i in range(10)]},
            {'name': 'enemy_score', 'type': 'digits', 'pattern': 'enemy_score_{}.npy', 'files': [f'enemy_score_{i}.npy' for i in range(10)]},
            {'name': 'goal_24p', 'type': 'single', 'file': 'goal_24p.npy'},

            {'name': 'travel', 'type': 'single', 'file': 'travel.npy'},
            {'name': 'out_of_bounds', 'type': 'single', 'file': 'out_of_bounds.npy'},
            {'name': 'clearance', 'type': 'single', 'file': 'clearance.npy'},
            {'name': 'turnover', 'type': 'single', 'file': 'turnover.npy'},
        ]
        
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/doubledunk"

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DunkGameState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Identify ball holder
        ball_holder = state.ball.holder
        
        # Clearance Logic Check (needed early for status indicators)
        p1_needs_clearance = jnp.logical_or(
            jnp.logical_and((ball_holder == PlayerID.PLAYER1_INSIDE), state.player1_inside.clearance_needed),
            jnp.logical_and((ball_holder == PlayerID.PLAYER1_OUTSIDE), state.player1_outside.clearance_needed)
        )

        # --- Prepare player data for sorting ---
        # 0: P1 Inside, 1: P1 Outside, 2: P2 Inside, 3: P2 Outside
        all_players_x = jnp.array([
            state.player1_inside.x, state.player1_outside.x,
            state.player2_inside.x, state.player2_outside.x,
        ])
        all_players_y = jnp.array([
            state.player1_inside.y, state.player1_outside.y,
            state.player2_inside.y, state.player2_outside.y,
        ])
        all_players_z = jnp.array([
            state.player1_inside.z, state.player1_outside.z,
            state.player2_inside.z, state.player2_outside.z,
        ])

        # Identify ball holder
        ball_holder = state.ball.holder
        
        # Determine Possession/Defense Role
        ball_holder_is_p1 = jnp.logical_or((ball_holder == PlayerID.PLAYER1_INSIDE), (ball_holder == PlayerID.PLAYER1_OUTSIDE))
        ball_holder_is_p2 = jnp.logical_or((ball_holder == PlayerID.PLAYER2_INSIDE), (ball_holder == PlayerID.PLAYER2_OUTSIDE))
        
        # P1 Team (0, 1) defends if P2 has ball
        # P2 Team (2, 3) defends if P1 has ball
        p1_defending = ball_holder_is_p2
        p2_defending = ball_holder_is_p1

        # --- Render players by visual Y position ---
        visual_ys = jnp.maximum(0, all_players_y - all_players_z)
        sort_indices = jnp.argsort(visual_ys)

        def render_player_body(i, current_raster):
            player_idx = sort_indices[i] # This is 0, 1, 2, or 3

            x = all_players_x[player_idx]
            visual_y = visual_ys[player_idx]
            z = all_players_z[player_idx]
            
            is_jumping = z > 0
            
            # Select Sprite Mask
            # 0: P1 Inside, 1: P1 Outside, 2: P2 Inside, 3: P2 Outside
            
            def get_p1_inside_mask():
                return jax.lax.select(
                    is_jumping, self.SHAPE_MASKS['player_inside_jump'],
                    jax.lax.select(p1_defending, self.SHAPE_MASKS['player_inside_guard'], self.SHAPE_MASKS['player_inside_1'])
                )
            
            def get_p1_outside_mask():
                return jax.lax.select(
                    is_jumping, self.SHAPE_MASKS['player_outside_jump'],
                    jax.lax.select(p1_defending, self.SHAPE_MASKS['player_outside_guard'], self.SHAPE_MASKS['player_outside_1'])
                )
                
            def get_p2_inside_mask():
                return jax.lax.select(
                    is_jumping, self.SHAPE_MASKS['enemy_inside_jump'],
                    jax.lax.select(p2_defending, self.SHAPE_MASKS['enemy_inside_guard'], self.SHAPE_MASKS['enemy_inside_1'])
                )

            def get_p2_outside_mask():
                return jax.lax.select(
                    is_jumping, self.SHAPE_MASKS['enemy_outside_jump'],
                    jax.lax.select(p2_defending, self.SHAPE_MASKS['enemy_outside_guard'], self.SHAPE_MASKS['enemy_outside_1'])
                )

            final_mask = jax.lax.switch(
                player_idx,
                [get_p1_inside_mask, get_p1_outside_mask, get_p2_inside_mask, get_p2_outside_mask]
            )

            return self.jr.render_at(current_raster, x, visual_y, final_mask)

        raster = jax.lax.fori_loop(0, 4, render_player_body, raster)

        # --- Render Status Indicators (OFF/DEF) ---
        def render_status_indicators(current_raster):
            # Determine complex possession including ball in flight
            # If no one holds the ball, check shooter or receiver
            
            p1_active = jnp.logical_or(ball_holder_is_p1, jnp.logical_and(
                (ball_holder == PlayerID.NONE), jnp.logical_or(jnp.logical_or(jnp.logical_or(
                    (state.ball.shooter == PlayerID.PLAYER1_INSIDE), (state.ball.shooter == PlayerID.PLAYER1_OUTSIDE)),
                    (state.ball.receiver == PlayerID.PLAYER1_INSIDE)), (state.ball.receiver == PlayerID.PLAYER1_OUTSIDE))
                )
            )
            
            p2_active = jnp.logical_or(ball_holder_is_p2, jnp.logical_and(
                (ball_holder == PlayerID.NONE), jnp.logical_or(jnp.logical_or(jnp.logical_or(
                    (state.ball.shooter == PlayerID.PLAYER2_INSIDE), (state.ball.shooter == PlayerID.PLAYER2_OUTSIDE)),
                    (state.ball.receiver == PlayerID.PLAYER2_INSIDE)), (state.ball.receiver == PlayerID.PLAYER2_OUTSIDE))
                )
            )
            
            # Logic: If P2 is active, P1 is DEF. Otherwise P1 is OFF.
            # This defaults to P1 OFF if both are false (start of game).
            p1_is_off = jnp.logical_not(p2_active)
            p2_is_off = jnp.logical_not(p1_is_off) # Complementary
            
            p1_mask = jax.lax.select(p1_is_off, self.SHAPE_MASKS['player_off'], self.SHAPE_MASKS['player_def'])
            p2_mask = jax.lax.select(p2_is_off, self.SHAPE_MASKS['enemy_off'], self.SHAPE_MASKS['enemy_def'])
            
            # Positions (Bottom of screen)
            y_pos = 185
            p1_x = 40
            p2_x = 100
            
            should_blink = (state.game_mode == GameMode.PLAY_SELECTION)
            is_penalty = jnp.logical_or(jnp.logical_or((state.game_mode == GameMode.TRAVEL_PENALTY), (state.game_mode == GameMode.OUT_OF_BOUNDS_PENALTY)), (state.game_mode == GameMode.CLEARANCE_PENALTY))
            # Added ~p1_needs_clearance to hide indicators during clearance warning
            is_visible = jnp.logical_and(jnp.logical_and(jnp.logical_not(is_penalty), jnp.logical_not(p1_needs_clearance)), jnp.logical_or(jnp.logical_not(should_blink), ((state.step_counter // 8) % 2 == 0)))

            current_raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(r, p1_x, y_pos, p1_mask),
                lambda r: r,
                current_raster
            )
            current_raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(r, p2_x, y_pos, p2_mask),
                lambda r: r,
                current_raster
            )
            
            return current_raster

        raster = render_status_indicators(raster)
        
        # --- Render Ball ---
        # Always render ball (even if held, to ensure visibility with new sprites)
        ball_mask = self.SHAPE_MASKS['ball']

        def render_ball_body(current_raster):
            holder_idx = jnp.clip(state.ball.holder - 1, 0, 3)
            holder_z = all_players_z[holder_idx]
            z_offset = jax.lax.select(state.ball.holder == PlayerID.NONE, 0, holder_z)

            ball_x = jnp.round(state.ball.x).astype(jnp.int32)
            ball_y = jnp.round(state.ball.y - z_offset).astype(jnp.int32)
            return self.jr.render_at(current_raster, ball_x, ball_y, ball_mask)

        raster = render_ball_body(raster)

        # --- Render Scores ---
        player_score_digits = self.jr.int_to_digits(state.scores.player, 2)
        enemy_score_digits = self.jr.int_to_digits(state.scores.enemy, 2)
        player_score_x = 40 
        enemy_score_x = 110 
        score_y = 10

        raster = jax.lax.cond(
            state.scores.player < 10,
            lambda r: self.jr.render_label_selective(r, player_score_x, score_y, player_score_digits, self.SHAPE_MASKS['player_score'], 1, 1, spacing=6),
            lambda r: self.jr.render_label_selective(r, player_score_x, score_y, player_score_digits, self.SHAPE_MASKS['player_score'], 0, 2, spacing=6),
            raster
        )

        raster = jax.lax.cond(
            state.scores.enemy < 10,
            lambda r: self.jr.render_label_selective(r, enemy_score_x, score_y, enemy_score_digits, self.SHAPE_MASKS['enemy_score'], 1, 1, spacing=6),
            lambda r: self.jr.render_label_selective(r, enemy_score_x, score_y, enemy_score_digits, self.SHAPE_MASKS['enemy_score'], 0, 2, spacing=6),
            raster
        )
        # Render the '24PTS' graphic under the player score
        raster = self.jr.render_at(raster, player_score_x - 6, score_y + 10, self.SHAPE_MASKS['goal_24p'])

        # First, always convert the base raster to an image
        final_image = self.jr.render_from_palette(raster, self.PALETTE)

        def apply_penalty_overlay(image, mask_name, timer):
            def render_mask(name):
                # Stamp the penalty text at the bottom
                mask = self.SHAPE_MASKS[name]
                
                # Convert text mask to RGB
                text_sprite_rgb = self.PALETTE[mask]
                
                # Create alpha mask for the text
                text_alpha_mask = (mask != self.jr.TRANSPARENT_ID)[..., None]

                # Position the text at the bottom
                text_x = (self.consts.WINDOW_WIDTH - mask.shape[1]) // 2
                text_y = 185

                # Get the slice from the image
                image_slice = jax.lax.dynamic_slice(
                    image,
                    (text_y, text_x, 0),
                    (mask.shape[0], mask.shape[1], 3)
                )

                # Blend the text onto the slice
                combined_slice = jnp.where(text_alpha_mask, text_sprite_rgb, image_slice)

                # Update the image with the blended slice
                return jax.lax.dynamic_update_slice(
                    image,
                    combined_slice,
                    (text_y, text_x, 0)
                )

            return jax.lax.cond(
                timer > 30,
                lambda: render_mask(mask_name),
                lambda: render_mask('turnover')
            )

        # Priority: TRAVEL > OUT OF BOUNDS > CLEARANCE
        is_travel_mode = (state.game_mode == GameMode.TRAVEL_PENALTY)
        is_oob_mode = (state.game_mode == GameMode.OUT_OF_BOUNDS_PENALTY)
        
        ball_holder = state.ball.holder
        p1_needs_clearance = jnp.logical_or(
            jnp.logical_and((ball_holder == PlayerID.PLAYER1_INSIDE), state.player1_inside.clearance_needed),
            jnp.logical_and((ball_holder == PlayerID.PLAYER1_OUTSIDE), state.player1_outside.clearance_needed)
        )
        
        is_clearance_mode = (state.game_mode == GameMode.CLEARANCE_PENALTY)
        effective_clearance_timer = jax.lax.select(is_clearance_mode, state.timers.clearance, 60)
        should_show_clearance = jnp.logical_or(p1_needs_clearance, is_clearance_mode)

        return jax.lax.cond(
            is_travel_mode,
            lambda x: apply_penalty_overlay(x, 'travel', state.timers.travel),
            lambda x: jax.lax.cond(
                is_oob_mode,
                lambda xx: apply_penalty_overlay(xx, 'out_of_bounds', state.timers.out_of_bounds),
                lambda xx: jax.lax.cond(
                    should_show_clearance,
                    lambda xxx: apply_penalty_overlay(xxx, 'clearance', effective_clearance_timer),
                    lambda xxx: xxx,
                    xx
                ),
                x
            ),
            final_image
        )