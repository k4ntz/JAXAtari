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

class DefensiveStrategy(IntEnum):
    LANE_DEFENSE = 0
    TIGHT_DEFENSE = 1

PICK_AND_ROLL = jnp.array([OffensiveAction.PASS, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])
GIVE_AND_GO = jnp.array([OffensiveAction.PASS, OffensiveAction.PASS, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])
MR_OUTSIDE_SHOOTS = jnp.array([OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT, OffensiveAction.JUMPSHOOT])

@chex.dataclass(frozen=True)
class DunkConstants:
    """Holds all static values for the game like screen dimensions, player speeds, colors, etc."""
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210
    BALL_SIZE: Tuple[int, int] = (3,3)
    JUMP_STRENGTH: int = 5
    PLAYER_MAX_SPEED: int = 2
    PLAYER_Y_MIN: int = 20
    PLAYER_Y_MAX: int = 130
    PLAYER_X_MIN: int  = 0
    PLAYER_X_MAX: int = 145
    PLAYER_WIDTH: int = 10                         
    PLAYER_HEIGHT: int = 30
    PLAYER_BARRIER: int = 10  
    BASKET_POSITION: Tuple[int,int] = (80,10)
    GRAVITY: int = 1
    AREA_3_POINT: Tuple[int,int,int] = (25, 135, 90) # (x_min, x_max, y_arc_connect) - needs a proper function to check if a point is in the 3-point area
    MAX_SCORE: int = 10
    DUNK_RADIUS: int = 18
    INSIDE_RADIUS: int = 50
    BLOCK_RADIUS: int = 14
    INSIDE_PLAYER_INSIDE_SHOT = 2
    OUTSIDE_PLAYER_OUTSIDE_SHOT = 2

@chex.dataclass(frozen=True)
class PlayerState:
    id: chex.Array # ID of the Player (see PlayerID) Practically a constant and is primarily used to check if the player is holding a ball for later purposes.
    #Position/Speed of Character
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
    game_mode: chex.Array           # Current mode (PLAY_SELECTION or IN_PLAY)
    strategy: OffensiveAction
    defensive_strategy: chex.Array
    offensive_strategy_step: chex.Array           # Tracks progress in p1 strategy (e.g., 1st, 2nd, 3rd button press)
    controlled_player_id: chex.Array
    offensive_action_cooldown: chex.Array
    enemy_reaction_timer: chex.Array
    last_enemy_actions: chex.Array
    travel_timer: chex.Array
    out_of_bounds_timer: chex.Array
    clearance_timer: chex.Array
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
STEAL_ACTIONS = {Action.UPFIRE}


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
            score_player=state.player_score.astype(jnp.int32),
            score_enemy=state.enemy_score.astype(jnp.int32),
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
        
        is_p2_holder = (holder == PlayerID.PLAYER2_INSIDE) | (holder == PlayerID.PLAYER2_OUTSIDE)
        
        p1_out_y = jax.lax.select(is_p2_holder, jnp.array(105, dtype=jnp.int32), jnp.array(115, dtype=jnp.int32))
        p2_out_y = jax.lax.select(is_p2_holder, jnp.array(115, dtype=jnp.int32), jnp.array(105, dtype=jnp.int32))

        return DunkGameState(
            player1_inside=PlayerState(id=1, x=100, y=60, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            player1_outside=PlayerState(id=2, x=75, y=p1_out_y, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            player2_inside=PlayerState(id=3, x=50, y=50, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            player2_outside=PlayerState(id=4, x=75, y=p2_out_y, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1, is_out_of_bounds=False, jumped_with_ball=False, triggered_travel=False, clearance_needed=False, triggered_clearance=False),
            # Start with a jump ball in the center: no holder and ball sits at the start position
            ball=BallState(x=50.0, y=110.0, vel_x=0.0, vel_y=0.0, holder=holder, target_x=0.0, target_y=0.0, landing_y=0.0, is_goal=False, shooter=PlayerID.NONE, receiver=PlayerID.NONE, shooter_pos_x=0, shooter_pos_y=0, missed_shot=False),
            player_score=jnp.array(0, dtype=jnp.int32),
            enemy_score=jnp.array(0, dtype=jnp.int32),
            step_counter=0,
            acceleration_counter=0,
            game_mode=GameMode.PLAY_SELECTION,
            strategy = GIVE_AND_GO,
            defensive_strategy = DefensiveStrategy.LANE_DEFENSE,
            offensive_strategy_step=0,
            controlled_player_id = PlayerID.PLAYER1_OUTSIDE,
            offensive_action_cooldown=0,
            enemy_reaction_timer=0,
            last_enemy_actions=jnp.array([Action.NOOP, Action.NOOP]),
            travel_timer=0,
            out_of_bounds_timer=0,
            clearance_timer=0,
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
        touched_bound = (updated_x <= constants.PLAYER_X_MIN) | (updated_x >= constants.PLAYER_X_MAX) | \
                        (updated_y <= constants.PLAYER_Y_MIN) | (updated_y >= constants.PLAYER_Y_MAX)

        # Clearance Check: Check if player is "inside". If not, they have cleared the ball.
        # Inside Zone definition based on scoring logic:
        # 1. Rectangular Zone: x=[25, 135], y <= 90
        in_rect_zone = (new_x >= 25) & (new_x <= 135) & (new_y <= 90)
        # 2. Elliptical Zone: Center(80, 80), Rx=55, Ry=45
        dx = new_x - 80.0
        dy = new_y - 80.0
        ellipse_val = (dx**2 / (55.0**2)) + (dy**2 / (45.0**2))
        in_ellipse_zone = (ellipse_val <= 1.0) & (new_y >= 80)
        
        is_inside = in_rect_zone | in_ellipse_zone
        is_outside = ~is_inside
        
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
        p1_outside_out_of_bounds = updated_p1_outside.is_out_of_bounds & (updated_p1_outside.id == ball_holder_id)
        p1_inside_out_of_bounds = updated_p1_inside.is_out_of_bounds & (updated_p1_inside.id == ball_holder_id)
        p1_out_of_bounds = p1_inside_out_of_bounds | p1_outside_out_of_bounds # if Player 1 triggered out of bounds

        # --- Reset Game State ---
        # Give ball to the team that didn't trigger out of bounds
        new_ball_holder = jax.lax.select(p1_out_of_bounds, PlayerID.PLAYER2_OUTSIDE, PlayerID.PLAYER1_OUTSIDE)

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
            out_of_bounds_timer=60,
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
            player_score=state.player_score,
            enemy_score=state.enemy_score,
            step_counter=state.step_counter,
        )

    def _handle_out_of_bounds_penalty(self, state: DunkGameState) -> DunkGameState:
        """Handles the out of bounds penalty freeze."""
        
        # Decrement timer
        new_timer = state.out_of_bounds_timer - 1
        timer_expired = new_timer <= 0

        # We need to know who held the ball to know who triggered it.
        # But the ball holder hasn't changed yet in 'state'.
        ball_holder_id = state.ball.holder
        
        p1_inside_out_of_bounds = state.player1_inside.is_out_of_bounds & (state.player1_inside.id == ball_holder_id)
        p1_outside_out_of_bounds = state.player1_outside.is_out_of_bounds & (state.player1_outside.id == ball_holder_id)
        p1_at_fault = p1_inside_out_of_bounds | p1_outside_out_of_bounds

        return jax.lax.cond(
            timer_expired,
            lambda s: self._resolve_penalty_and_reset(s, p1_at_fault),
            lambda s: s.replace(out_of_bounds_timer=new_timer),
            state
        )

    def _update_player_z(self, player: PlayerState, constants: DunkConstants, ball_hold_id: int) -> PlayerState:
        """Applies Z-axis physics (jumping and gravity) to a player."""
        has_ball = (ball_hold_id == player.id) #check if the player has the ball
        new_z = player.z + player.vel_z
        new_vel_z = player.vel_z - constants.GRAVITY
        has_landed = new_z <= 0 #check if the player is back on the ground
        jump_start_with_ball = (player.z == 0) & has_ball #check if the player starts a jump while having the ball
        new_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_z)
        new_vel_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_vel_z)
        has_triggered_travel = has_landed & player.jumped_with_ball # True if the player lands while having the ball at the start of the jump
        # update PlayerState value jumped_with_ball
        updated_jumped_with_ball = jax.lax.select(jump_start_with_ball, ~has_landed,
                                    jax.lax.select(~has_landed,has_ball & player.jumped_with_ball, False))
        # update PlayerState with new height and whether or not they started/finished the jump with the ball
        return player.replace(z=new_z, vel_z=new_vel_z, jumped_with_ball=updated_jumped_with_ball, triggered_travel=has_triggered_travel)

    def _update_players_z(self, state: DunkGameState) -> DunkGameState:
        """Applies Z-axis physics for all players."""
        ball_holder_id = state.ball.holder
        
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        
        updated_players = jax.vmap(self._update_player_z, in_axes=(0, None, None))(players, self.constants, ball_holder_id)
        
        updated_p1_inside, updated_p1_outside, updated_p2_inside, updated_p2_outside = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

        # check if any players triggered the travel rule
        travel_triggered = updated_p1_outside.triggered_travel | updated_p1_inside.triggered_travel


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
            travel_timer=60
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
            landing_y=float(self.constants.PLAYER_Y_MIN+20), # Ground level
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
            action = jax.lax.select(want_up & want_left, Action.UPLEFT, action)
            action = jax.lax.select(want_up & want_right, Action.UPRIGHT, action)
            action = jax.lax.select(want_down & want_left, Action.DOWNLEFT, action)
            action = jax.lax.select(want_down & want_right, Action.DOWNRIGHT, action)
            
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
        p1_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE) | (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        p2_has_ball = (state.ball.holder == PlayerID.PLAYER2_INSIDE) | (state.ball.holder == PlayerID.PLAYER2_OUTSIDE)
        
        defensive_strat = state.defensive_strategy
        basket_x, basket_y = self.constants.BASKET_POSITION

        # --- Defensive Logic (Target Calculation) ---
        
        # P1 Defensive Targets (if P1 is defending)
        p1_in_target_lane_x = (state.player2_inside.x + state.player2_outside.x) // 2
        p1_in_target_lane_y = (state.player2_inside.y + state.player2_outside.y) // 2
        p1_out_target_lane_x = state.player2_outside.x
        p1_out_target_lane_y = state.player2_outside.y
        
        p1_in_target_tight_x = state.player2_inside.x
        p1_in_target_tight_y = state.player2_inside.y
        p1_out_target_tight_x = state.player2_outside.x
        p1_out_target_tight_y = state.player2_outside.y
        
        p1_in_def_x = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p1_in_target_lane_x, p1_in_target_tight_x)
        p1_in_def_y = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p1_in_target_lane_y, p1_in_target_tight_y)
        p1_out_def_x = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p1_out_target_lane_x, p1_out_target_tight_x)
        p1_out_def_y = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p1_out_target_lane_y, p1_out_target_tight_y)
        
        p1_in_def_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, p1_in_def_x, p1_in_def_y)
        p1_out_def_action = get_move_to_target(state.player1_outside.x, state.player1_outside.y, p1_out_def_x, p1_out_def_y)

        # P2 Defensive Targets (if P2 is defending)
        p2_in_target_lane_x = (state.player1_inside.x + state.player1_outside.x) // 2
        p2_in_target_lane_y = (state.player1_inside.y + state.player1_outside.y) // 2
        p2_out_target_lane_x = state.player1_outside.x
        p2_out_target_lane_y = state.player1_outside.y

        p2_in_target_tight_x = state.player1_inside.x
        p2_in_target_tight_y = state.player1_inside.y
        p2_out_target_tight_x = state.player1_outside.x
        p2_out_target_tight_y = state.player1_outside.y

        p2_in_def_x = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p2_in_target_lane_x, p2_in_target_tight_x)
        p2_in_def_y = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p2_in_target_lane_y, p2_in_target_tight_y)
        p2_out_def_x = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p2_out_target_lane_x, p2_out_target_tight_x)
        p2_out_def_y = jax.lax.select(defensive_strat == DefensiveStrategy.LANE_DEFENSE, p2_out_target_lane_y, p2_out_target_tight_y)

        p2_in_def_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, p2_in_def_x, p2_in_def_y)
        p2_out_def_action = get_move_to_target(state.player2_outside.x, state.player2_outside.y, p2_out_def_x, p2_out_def_y)

        # --- Teammate AI (Offensive/Random vs Defensive) ---
        is_p1_inside_teammate_ai = ~is_p1_inside_controlled
        is_p1_outside_teammate_ai = ~is_p1_outside_controlled

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
        p1_is_far = (p1_dist_to_basket_x > 20) | (p1_dist_to_basket_y > 80) # 40x80 area
        p1_return_to_basket_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, basket_x, basket_y)
        
        p1_inside_action = jax.lax.select(
            is_p1_inside_teammate_ai,
            jax.lax.select(
                ball_is_free,
                p1_in_chase,
                jax.lax.select(
                    p2_has_ball, 
                    p1_in_def_action, 
                    jax.lax.select(p1_is_far, p1_return_to_basket_action, p1_off_action)
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
        p2_is_far = (p2_dist_to_basket_x > 20) | (p2_dist_to_basket_y > 40) # 40x80 area
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
            go_left = (dist_left < dist_right) & (dist_left < dist_down)
            go_right = (dist_right <= dist_left) & (dist_right < dist_down)
            # go_down is implicit else
            
            tx = jax.lax.select(go_left, 20.0, jax.lax.select(go_right, 140.0, px_f))
            ty = jax.lax.select(go_left | go_right, py_f, boundary_y + 10.0)
            
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

        # --- Enemy Reaction Time Logic ---
        use_last_action = state.enemy_reaction_timer > 0
        
        final_p2_inside_action = jax.lax.select(use_last_action, state.last_enemy_actions[0], p2_inside_action)
        final_p2_outside_action = jax.lax.select(use_last_action, state.last_enemy_actions[1], p2_outside_action)
        
        new_timer = jax.lax.select(use_last_action, state.enemy_reaction_timer - 1, 6) # Reset to 6 frames (~0.1s)
        new_last_actions = jax.lax.select(use_last_action, state.last_enemy_actions, jnp.array([p2_inside_action, p2_outside_action]))

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
        is_pass_step = (state.strategy[state.offensive_strategy_step] == OffensiveAction.PASS)
        
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        def check_passing(pid, p_action):
            is_passing = (state.offensive_action_cooldown == 0) & is_pass_step & (ball_state.holder == pid) & jnp.any(jnp.asarray(p_action) == jnp.asarray(list(_PASS_ACTIONS)))
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
        is_shoot_step = (state.strategy[state.offensive_strategy_step] == OffensiveAction.JUMPSHOOT)
        
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        def check_shooting(pid, p_z, p_action):
            is_shooting = (state.offensive_action_cooldown == 0) & is_shoot_step & (p_z != 0) & (ball_state.holder == pid) & jnp.any(jnp.asarray(p_action) == jnp.asarray(list(_SHOOT_ACTIONS)))
            return is_shooting

        shooting_flags = jax.vmap(check_shooting)(player_ids, players.z, actions_stacked)
        is_shooting = jnp.any(shooting_flags)
        shooter_idx = jnp.argmax(shooting_flags)
        shooter = player_ids[shooter_idx]
        shooter_x = players.x[shooter_idx]
        shooter_y = players.y[shooter_idx]
        shooter_z = players.z[shooter_idx]

        is_inside_shooting = (shooter == PlayerID.PLAYER1_INSIDE) | (shooter == PlayerID.PLAYER2_INSIDE)
        is_outside_shooting = (shooter == PlayerID.PLAYER1_OUTSIDE) | (shooter == PlayerID.PLAYER2_OUTSIDE)

        key, offset_key_x = random.split(key)

        shooter_pos = jnp.array([shooter_x, shooter_y], dtype=jnp.float32)
        basket_pos = jnp.array([self.constants.BASKET_POSITION[0], self.constants.BASKET_POSITION[1]], dtype=jnp.float32)
        dist_to_basket = jnp.sqrt(jnp.sum((shooter_pos - basket_pos) ** 2))
        is_inside = dist_to_basket < self.constants.INSIDE_RADIUS
        shot_bonus = jax.lax.select(is_inside & is_inside_shooting, 2, jax.lax.select(~is_inside & is_outside_shooting, 2, -2))

        offset_x = random.uniform(offset_key_x, shape=(), minval=-10 + shot_bonus, maxval=10 - shot_bonus)
        is_goal = (offset_x >= -5) & (offset_x <= 5)

        target_pos = basket_pos + jnp.array([offset_x, 0.0])

        shoot_direction = target_pos - shooter_pos
        shoot_norm = jnp.sqrt(jnp.sum(shoot_direction**2))
        shoot_safe_norm = jnp.where(shoot_norm == 0, 1.0, shoot_norm)
        shoot_speed = 8.0
        shoot_vel = (shoot_direction / shoot_safe_norm) * shoot_speed

        # Basic blocking
        def check_blocking():
            # opponents (if shooter is P1, opponents are P2; if shooter is P2, opponents are P1)
            is_shooter_p1 = (shooter == PlayerID.PLAYER1_INSIDE) | (shooter == PlayerID.PLAYER1_OUTSIDE)
            
            opp_indices = jax.lax.select(is_shooter_p1, jnp.array([2, 3]), jnp.array([0, 1]))
            opp_xs = players.x[opp_indices]
            opp_ys = players.y[opp_indices]
            opp_zs = players.z[opp_indices]
            opp_ids = player_ids[opp_indices]

            dists = jnp.sqrt((opp_xs - shooter_x)**2 + (opp_ys - shooter_y)**2)
            can_blocks = (opp_zs > 0) & (dists < self.constants.BLOCK_RADIUS)
            
            blocked_by = jax.lax.select(can_blocks[0], opp_ids[0], jax.lax.select(can_blocks[1], opp_ids[1], PlayerID.NONE))
            return blocked_by

        blocked_by = jax.lax.cond(is_shooting, check_blocking, lambda: PlayerID.NONE)

        is_dunk = (dist_to_basket < self.constants.DUNK_RADIUS) & (shooter_z > 0)

        def make_shot(b):
            b = b.replace(x=shooter_x.astype(jnp.float32), y=shooter_y.astype(jnp.float32), vel_x=shoot_vel[0], vel_y=shoot_vel[1], holder=PlayerID.NONE, target_x=target_pos[0], target_y=target_pos[1], is_goal=is_goal, shooter=shooter, receiver=PlayerID.NONE, shooter_pos_x=shooter_x.astype(jnp.int32), shooter_pos_y=shooter_y.astype(jnp.int32))
            b = jax.lax.cond(blocked_by != PlayerID.NONE, lambda bb: bb.replace(holder=blocked_by, vel_x=0.0, vel_y=0.0, is_goal=False, shooter=PlayerID.NONE), lambda bb: bb, b)
            b = jax.lax.cond(is_dunk, lambda bb: bb.replace(is_goal=True, target_x=basket_pos[0], target_y=basket_pos[1]), lambda bb: bb, b)
            return b

        new_ball_state = jax.lax.cond(
            is_shooting,
            make_shot,
            lambda b: b,
            ball_state
        )
        step_increment = jax.lax.select(is_shooting, 1, 0)
        return new_ball_state, key, step_increment, is_shooting

        # Determine whether this shot should be a dunk (inside player jumping near basket)

        is_dunk = (dist_to_basket < self.constants.DUNK_RADIUS) & (shooter_z > 0)

        def make_shot(b):
            # If blocked by opponent, possession goes to blocker
            b = b.replace(x=shooter_x.astype(jnp.float32), y=shooter_y.astype(jnp.float32), vel_x=shoot_vel[0], vel_y=shoot_vel[1], holder=PlayerID.NONE, target_x=target_pos[0], target_y=target_pos[1], is_goal=is_goal, shooter=shooter, receiver=PlayerID.NONE, shooter_pos_x=shooter_x.astype(jnp.int32), shooter_pos_y=shooter_y.astype(jnp.int32))
            b = jax.lax.cond(blocked_by != PlayerID.NONE, lambda bb: bb.replace(holder=blocked_by, vel_x=0.0, vel_y=0.0, is_goal=False, shooter=PlayerID.NONE), lambda bb: bb, b)
            # If dunk, bump is_goal to True and make target the rim
            b = jax.lax.cond(is_dunk, lambda bb: bb.replace(is_goal=True, target_x=basket_pos[0], target_y=basket_pos[1]), lambda bb: bb, b)
            return b

        new_ball_state = jax.lax.cond(
            is_shooting,
            make_shot,
            lambda b: b,
            ball_state
        )
        step_increment = jax.lax.select(is_shooting, 1, 0)
        return new_ball_state, key, step_increment, is_shooting

    def _handle_stealing(self, state: DunkGameState, actions: Tuple[int, ...]) -> BallState:
        """Handles the logic for stealing the ball."""
        ball_state = state.ball
        
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)
        player_ids = jnp.array([PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE])

        def check_steal(player, action, pid):
            is_trying_to_steal = jnp.any(jnp.asarray(action) == jnp.asarray(list(STEAL_ACTIONS)))
            dist_sq = (player.x - ball_state.x)**2 + (player.y - ball_state.y)**2
            is_close_to_ball = dist_sq < 2401.0 # 49^2
            
            # Can only steal if opponent has ball
            # Team 1: IDs 1, 2. Team 2: IDs 3, 4.
            is_p1_team = (pid <= 2)
            holder_is_p1_team = (ball_state.holder == PlayerID.PLAYER1_INSIDE) | (ball_state.holder == PlayerID.PLAYER1_OUTSIDE)
            holder_is_p2_team = (ball_state.holder == PlayerID.PLAYER2_INSIDE) | (ball_state.holder == PlayerID.PLAYER2_OUTSIDE)
            
            can_steal = (is_p1_team & holder_is_p2_team) | (~is_p1_team & holder_is_p1_team)
            
            return is_trying_to_steal & is_close_to_ball & can_steal

        steal_flags = jax.vmap(check_steal)(players, actions_stacked, player_ids)
        
        is_stealing = jnp.any(steal_flags)
        stealer_idx = jnp.argmax(steal_flags)
        stealer_id = player_ids[stealer_idx]

        new_ball_state = jax.lax.cond(
            is_stealing,
            lambda b: b.replace(holder=stealer_id, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            ball_state
        )
        return new_ball_state

    def _handle_jump(self, state: DunkGameState, player: PlayerState, action: int, constants: DunkConstants) -> chex.Array:
        """Calculates the vertical impulse for a jump."""
        is_jump_step = (state.strategy[state.offensive_strategy_step] == OffensiveAction.JUMPSHOOT) & (player.z == 0)
        can_jump = (state.offensive_action_cooldown == 0) & is_jump_step & jnp.any(jnp.asarray(action) == jnp.asarray(list(_JUMP_ACTIONS)))
        vel_z = jax.lax.select(can_jump, constants.JUMP_STRENGTH, jnp.array(0, dtype=jnp.int32))
        new_vel_z = jax.lax.select(vel_z > 0, vel_z, player.vel_z)
        did_jump = jax.lax.select(can_jump, 1, 0)

        # Clearance Penalty Check
        has_ball = (state.ball.holder == player.id)
        triggered_clearance = can_jump & has_ball & player.clearance_needed

        return player.replace(vel_z=new_vel_z, triggered_clearance=triggered_clearance), did_jump

    def _handle_offense_actions(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[DunkGameState, chex.PRNGKey, chex.Array]:
        """Handles offensive actions: passing and shooting."""
        # Passing
        ball_state_after_pass, pass_inc, did_pass = self._handle_passing(state, actions)
        state = state.replace(ball=ball_state_after_pass)
        state = state.replace(offensive_action_cooldown=jax.lax.select(did_pass, 6, state.offensive_action_cooldown))

        # Shooting
        ball_state_after_shot, key, shot_inc, did_shoot = self._handle_shooting(state, actions, key)
        state = state.replace(ball=ball_state_after_shot)
        state = state.replace(offensive_action_cooldown=jax.lax.select(did_shoot, 6, state.offensive_action_cooldown))

        step_increment = pass_inc + shot_inc
        return state, key, step_increment

    def _handle_defense_actions(self, state: DunkGameState, actions: Tuple[int, ...]) -> DunkGameState:
        """Handles defensive actions: stealing."""
        ball_state_after_steal = self._handle_stealing(state, actions)
        state = state.replace(ball=ball_state_after_steal)
        return state

    def _handle_interactions(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[DunkGameState, chex.PRNGKey]:
        """Handles all player interactions: jump, pass, shoot, steal."""
        players = jax.tree_util.tree_map(lambda *args: jnp.stack(args), state.player1_inside, state.player1_outside, state.player2_inside, state.player2_outside)
        actions_stacked = jnp.stack(actions)

        updated_players, jumped_flags = jax.vmap(self._handle_jump, in_axes=(None, 0, 0, None))(state, players, actions_stacked, self.constants)
        
        updated_p1_inside, updated_p1_outside, updated_p2_inside, updated_p2_outside = [jax.tree_util.tree_map(lambda x: x[i], updated_players) for i in range(4)]

        did_jump = jnp.max(jumped_flags)

        state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )
        state = state.replace(offensive_action_cooldown=jax.lax.select(did_jump > 0, 6, state.offensive_action_cooldown))

        # 2. Handle ball actions
        state, key, offense_increment = self._handle_offense_actions(state, actions, key)
        state = self._handle_defense_actions(state, actions)

        # 3. Update offensive_strategy_step
        new_offensive_strategy_step = jnp.minimum(state.offensive_strategy_step + offense_increment, len(state.strategy)-1)
        
        # 4. Print if changed
        # We use jax.lax.cond to ensure we only print when an action actually occurred
        jax.lax.cond(
            offense_increment > 0,
            lambda x: jax.debug.print("Play Step: {}", x),
            lambda x: None,
            new_offensive_strategy_step
        )

        return state.replace(offensive_strategy_step=new_offensive_strategy_step), key

    def _update_ball(self, state: DunkGameState) -> DunkGameState:
        """Handles ball movement, goals, misses, catches, and possession changes."""
        ball_in_flight = (state.ball.holder == PlayerID.NONE)
        dist_to_target = jnp.sqrt((state.ball.x - state.ball.target_x)**2 + (state.ball.y - state.ball.target_y)**2)
        is_already_falling = (state.ball.vel_x == 0) & jnp.isclose(state.ball.vel_y, 2.0)
        reached_target = ball_in_flight & (dist_to_target < 5.0) & ~is_already_falling
        is_goal_scored = reached_target & state.ball.is_goal

        def on_goal(s):
            key, reset_key = random.split(s.key)
            is_p1_scorer = (s.ball.shooter == PlayerID.PLAYER1_INSIDE) | (s.ball.shooter == PlayerID.PLAYER1_OUTSIDE)

            # --- 2 vs 3 Point Logic ---
            # Based on the background generation script geometry:
            # 1. Rectangular Zone: x=[25, 135], y <= 90
            # 2. Elliptical Zone: Center(80, 80), Rx=55, Ry=45 (approx due to perspective)
            
            sx = s.ball.shooter_pos_x
            sy = s.ball.shooter_pos_y

            # Check 1: Rectangular Key Area
            # The script draws vertical lines at x=25 and x=135 down to y=90
            in_rect_zone = (sx >= 25) & (sx <= 135) & (sy <= 90)

            # Check 2: Elliptical Top of Key
            # Equation: ((x-h)/rx)^2 + ((y-k)/ry)^2 <= 1
            # Center (h,k) = (80, 80)
            dx = sx - 80.0
            dy = sy - 80.0
            
            # We use float division for the ellipse calculation
            ellipse_val = (dx**2 / (55.0**2)) + (dy**2 / (45.0**2))
            
            # We only care about the ellipse part that extends below the center (y >= 80)
            in_ellipse_zone = (ellipse_val <= 1.0) & (sy >= 80)

            # A shot is 2 points if it is in EITHER zone. Otherwise, it's a 3-pointer.
            is_2_point = in_rect_zone | in_ellipse_zone
            points = jax.lax.select(is_2_point, 2, 3)

            # --- Score Updates ---
            new_player_score = s.player_score + points * is_p1_scorer
            new_enemy_score = s.enemy_score + points * (1 - is_p1_scorer)

            # --- Reset Game State ---
            # Give ball to the team that got scored on
            new_ball_holder = jax.lax.select(is_p1_scorer, PlayerID.PLAYER2_OUTSIDE, PlayerID.PLAYER1_OUTSIDE)

            # We recreate the initial state but preserve scores and step counter
            new_state = self._init_state(reset_key, holder=new_ball_holder).replace(
                player_score=new_player_score, 
                enemy_score=new_enemy_score, 
                step_counter=s.step_counter
            )
            
            return new_state

        def continue_play(s):
            # Handle miss
            is_miss = reached_target & ~s.ball.is_goal
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
            is_falling = (b_state.vel_x == 0) & jnp.isclose(b_state.vel_y, 2.0)
            new_ball_x = b_state.x + b_state.vel_x
            new_ball_y = b_state.y + b_state.vel_y
            has_landed = is_falling & (new_ball_y >= b_state.landing_y)
            final_y = jax.lax.select(has_landed, b_state.landing_y, new_ball_y)
            final_vel_y = jax.lax.select(has_landed, 0.0, b_state.vel_y)
            b_state = jax.lax.cond(
                (b_state.holder == PlayerID.NONE),
                lambda b: b.replace(x=new_ball_x, y=final_y, vel_y=final_vel_y),
                lambda b: b,
                b_state
            )

            # Handle interceptions and catches
            catch_radius_sq = 49.0 # optimal value: tried different values
            ball_in_flight_after_physics = (b_state.holder == PlayerID.NONE)

            def check_catch_flag(px, py):
                dist_sq = (b_state.x - px)**2 + (b_state.y - py)**2
                return ball_in_flight_after_physics & (dist_sq < catch_radius_sq)

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
                is_rebound = (shooter_id != PlayerID.NONE) & curr_ball.missed_shot
                
                shooter_team_p1 = (shooter_id == PlayerID.PLAYER1_INSIDE) | (shooter_id == PlayerID.PLAYER1_OUTSIDE)
                catcher_team_p1 = (pid == PlayerID.PLAYER1_INSIDE) | (pid == PlayerID.PLAYER1_OUTSIDE)
                is_defensive_rebound = is_rebound & (shooter_team_p1 != catcher_team_p1)
                
                # Update ball
                new_ball = curr_ball.replace(holder=pid, vel_x=0.0, vel_y=0.0, receiver=PlayerID.NONE, shooter=PlayerID.NONE, missed_shot=False)
                
                # Update player state using vmap
                players_stacked_catch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), curr_state.player1_inside, curr_state.player1_outside, curr_state.player2_inside, curr_state.player2_outside)
                
                def update_clearance(player, p_id):
                    should_set = (p_id == pid) & is_defensive_rebound
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
            is_held = (b_state.holder >= PlayerID.PLAYER1_INSIDE) & (b_state.holder <= PlayerID.PLAYER2_OUTSIDE)
            def update_held_pos(b):
                idx = b.holder - 1
                return b.replace(x=players_stacked_new.x[idx].astype(jnp.float32), y=players_stacked_new.y[idx].astype(jnp.float32))
            
            b_state = jax.lax.cond(is_held, update_held_pos, lambda b: b, b_state)

            # Update controlled player
            is_p1_team_holder = (b_state.holder == PlayerID.PLAYER1_INSIDE) | (b_state.holder == PlayerID.PLAYER1_OUTSIDE)
            new_controlled_player_id = jax.lax.select(is_p1_team_holder, b_state.holder, s.controlled_player_id)
            
            return s.replace(ball=b_state, controlled_player_id=new_controlled_player_id)

        final_state = jax.lax.cond(is_goal_scored, on_goal, continue_play, state)
        return final_state

    def _handle_play_selection(self, state: DunkGameState, action: int) -> DunkGameState:
        """Handles the play selection mode."""
        p1_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE) | (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        
        key, strat_key = random.split(state.key)

        # Enemy choices
        random_offensive_idx = random.randint(strat_key, shape=(), minval=0, maxval=3) # 0, 1, 2
        enemy_offensive_strategy = jax.lax.switch(random_offensive_idx, [lambda: PICK_AND_ROLL, lambda: GIVE_AND_GO, lambda: MR_OUTSIDE_SHOOTS])
        
        random_defensive_idx = random.randint(strat_key, shape=(), minval=0, maxval=2) # 0, 1
        enemy_defensive_strategy = random_defensive_idx

        def start_game(s, selected_off, selected_def):
            # When a strategy is selected, we reset the game to its initial state
            # but keep the selected strategy and switch to IN_PLAY mode.
            # We also keep the current key.
            return s.replace(
                strategy=selected_off,
                defensive_strategy=selected_def,
                game_mode=GameMode.IN_PLAY,
                key=key
            )

        is_up = (action == Action.UPFIRE)
        is_down = (action == Action.DOWNFIRE)
        is_right = (action == Action.RIGHTFIRE)

        # Case 1: P1 Has Ball (User chooses Offense, Enemy chooses Defense)
        # User choices: UP=PICK_AND_ROLL, DOWN=GIVE_AND_GO, RIGHT=MR_OUTSIDE_SHOOTS
        off_p1 = jax.lax.select(is_up, PICK_AND_ROLL, 
                   jax.lax.select(is_down, GIVE_AND_GO, 
                     jax.lax.select(is_right, MR_OUTSIDE_SHOOTS, state.strategy)))
        
        valid_input_p1_off = is_up | is_down | is_right
        
        state_p1_ball = jax.lax.cond(
            valid_input_p1_off,
            lambda s: start_game(s, off_p1, enemy_defensive_strategy),
            lambda s: s,
            state
        )

        # Case 2: P2 Has Ball (Enemy chooses Offense, User chooses Defense)
        # User choices: UP=LANE_DEFENSE, DOWN=TIGHT_DEFENSE, RIGHT=LANE_DEFENSE (Default)
        def_p1 = jax.lax.select(is_up, DefensiveStrategy.LANE_DEFENSE,
                   jax.lax.select(is_down, DefensiveStrategy.TIGHT_DEFENSE, 
                     jax.lax.select(is_right, DefensiveStrategy.LANE_DEFENSE, state.defensive_strategy)))
        
        valid_input_p1_def = is_up | is_down # Right maps to something or ignored? Let's include Right as default/LANE for now or strict?
        # Prompt says: "whoever has the ball (User or enemy) must choose an offensive strategy and the other team chooses a defensive strategy"
        # I'll stick to Up/Down for defense as I only have 2 defensive strategies.

        state_p2_ball = jax.lax.cond(
             valid_input_p1_def,
             lambda s: start_game(s, enemy_offensive_strategy, def_p1),
             lambda s: s,
             state
        )

        return jax.lax.cond(p1_has_ball, lambda: state_p1_ball, lambda: state_p2_ball)

    def _handle_travel_penalty(self, state: DunkGameState) -> DunkGameState:
        """Handles the travel penalty freeze."""
        
        # Decrement timer
        new_timer = state.travel_timer - 1
        timer_expired = new_timer <= 0

        p1_triggered = state.player1_inside.triggered_travel | state.player1_outside.triggered_travel

        return jax.lax.cond(
            timer_expired,
            lambda s: self._resolve_penalty_and_reset(s, p1_triggered),
            lambda s: s.replace(travel_timer=new_timer),
            state
        )

    def _handle_clearance_penalty(self, state: DunkGameState) -> DunkGameState:
        """Handles the clearance penalty freeze."""
        
        # Decrement timer
        new_timer = state.clearance_timer - 1
        timer_expired = new_timer <= 0

        p1_triggered = state.player1_inside.triggered_clearance | state.player1_outside.triggered_clearance

        return jax.lax.cond(
            timer_expired,
            lambda s: self._resolve_penalty_and_reset(s, p1_triggered),
            lambda s: s.replace(clearance_timer=new_timer),
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
            s = s.replace(offensive_action_cooldown=jnp.maximum(0, s.offensive_action_cooldown - 1))

            # 1. Determine actions for all players
            actions, key, new_timer, new_last_actions = self._handle_player_actions(s, action, s.key)
            s = s.replace(enemy_reaction_timer=new_timer, last_enemy_actions=new_last_actions)

            # 2. Update player XY movement
            s = self._update_players_xy(s, actions)

            # 3. Handle interactions (jump, pass, shoot, steal)
            s, key = self._handle_interactions(s, actions, key)
            
            # Check for Clearance Penalty Trigger (after jump)
            p1_clearance = s.player1_inside.triggered_clearance | s.player1_outside.triggered_clearance
            p2_clearance = s.player2_inside.triggered_clearance | s.player2_outside.triggered_clearance
            clearance_triggered = p1_clearance | p2_clearance
            
            s = jax.lax.cond(
                clearance_triggered,
                lambda s_: s_.replace(game_mode=GameMode.CLEARANCE_PENALTY, clearance_timer=60),
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
            lambda s: s, # If not in play, do nothing
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
        is_max_score_reached = (state.player_score >= self.constants.MAX_SCORE) | (state.enemy_score >= self.constants.MAX_SCORE)
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
            
            # Team 1 (Player) - Blue/Black
            {'name': 'player_light', 'type': 'group', 'files': [f'player_light_{i}.npy' for i in range(10)]},
            {'name': 'player_dark', 'type': 'group', 'files': [f'player_dark_{i}.npy' for i in range(10)]},
            {'name': 'player_light_no_ball', 'type': 'single', 'file': 'player_light_no_ball.npy'},
            {'name': 'player_dark_no_ball', 'type': 'single', 'file': 'player_dark_no_ball.npy'},

            # Team 2 (Enemy) - Red/White
            {'name': 'enemy_light', 'type': 'group', 'files': [f'enemy_light_{i}.npy' for i in range(10)]},
            {'name': 'enemy_dark', 'type': 'group', 'files': [f'enemy_dark_{i}.npy' for i in range(10)]},
            {'name': 'enemy_light_no_ball', 'type': 'single', 'file': 'enemy_light_no_ball.npy'},
            {'name': 'enemy_dark_no_ball', 'type': 'single', 'file': 'enemy_dark_no_ball.npy'},

            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
            {'name': 'player_arrow', 'type': 'single', 'file': 'player_arrow.npy'},
            {'name': 'score', 'type': 'digits', 'pattern': 'score_{}.npy', 'files': [f'score_{i}.npy' for i in range(21)]},
            {'name': 'play_selection', 'type': 'single', 'file': 'play_selection.npy'},
            {'name': 'travel', 'type': 'single', 'file': 'travel.npy'},
            {'name': 'out_of_bounds', 'type': 'single', 'file': 'out_of_bounds.npy'},
            {'name': 'clearance', 'type': 'single', 'file': 'clearance.npy'},
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
        all_players_anim_frame = jnp.array([
            state.player1_inside.animation_frame, state.player1_outside.animation_frame,
            state.player2_inside.animation_frame, state.player2_outside.animation_frame,
        ])

        # Identify ball holder
        ball_holder = state.ball.holder
        has_ball = jnp.array([
            ball_holder == PlayerID.PLAYER1_INSIDE,
            ball_holder == PlayerID.PLAYER1_OUTSIDE,
            ball_holder == PlayerID.PLAYER2_INSIDE,
            ball_holder == PlayerID.PLAYER2_OUTSIDE,
        ])

        # --- Render players by visual Y position ---
        visual_ys = all_players_y - all_players_z
        sort_indices = jnp.argsort(visual_ys)

        def render_player_body(i, current_raster):
            player_idx = sort_indices[i] # This is 0, 1, 2, or 3

            x = all_players_x[player_idx]
            visual_y = visual_ys[player_idx]
            anim_frame = all_players_anim_frame[player_idx]
            p_has_ball = has_ball[player_idx]

            # --- Select Masks based on Player Index ---
            # 0: P1 Inside (Dark)
            # 1: P1 Outside (Light)
            # 2: P2 Inside (Dark)
            # 3: P2 Outside (Light)

            # Retrieve all possible masks for this specific animation frame
            p_dark_ball = self.SHAPE_MASKS['player_dark'][anim_frame]
            p_light_ball = self.SHAPE_MASKS['player_light'][anim_frame]
            e_dark_ball = self.SHAPE_MASKS['enemy_dark'][anim_frame]
            e_light_ball = self.SHAPE_MASKS['enemy_light'][anim_frame]

            # Retrieve all possible masks for no ball
            p_dark_no = self.SHAPE_MASKS['player_dark_no_ball']
            p_light_no = self.SHAPE_MASKS['player_light_no_ball']
            e_dark_no = self.SHAPE_MASKS['enemy_dark_no_ball']
            e_light_no = self.SHAPE_MASKS['enemy_light_no_ball']

            # Use jax.lax.switch to pick the correct sprite set based on player_idx
            mask_with_ball = jax.lax.switch(
                player_idx,
                [
                    lambda: p_dark_ball,  # 0: P1 Inside
                    lambda: p_light_ball, # 1: P1 Outside
                    lambda: e_dark_ball,  # 2: P2 Inside
                    lambda: e_light_ball  # 3: P2 Outside
                ]
            )

            mask_no_ball = jax.lax.switch(
                player_idx,
                [
                    lambda: p_dark_no,    # 0: P1 Inside
                    lambda: p_light_no,   # 1: P1 Outside
                    lambda: e_dark_no,    # 2: P2 Inside
                    lambda: e_light_no    # 3: P2 Outside
                ]
            )

            final_mask = jax.lax.select(p_has_ball, mask_with_ball, mask_no_ball)

            return self.jr.render_at(current_raster, x, visual_y, final_mask)

        raster = jax.lax.fori_loop(0, 4, render_player_body, raster)

        # --- Render Controlled Player Arrow ---        
        def render_arrow_body(current_raster):
            controlled_x = all_players_x[state.controlled_player_id - 1]
            controlled_visual_y = visual_ys[state.controlled_player_id - 1]
            arrow_mask = self.SHAPE_MASKS['player_arrow']
            arrow_height = arrow_mask.shape[0]
            
            arrow_y = controlled_visual_y - arrow_height
            player_width = 10 
            arrow_width = arrow_mask.shape[1]
            arrow_x = controlled_x + (player_width // 2) - (arrow_width // 2) + 2

            return self.jr.render_at(current_raster, arrow_x, arrow_y, arrow_mask)

        raster = jax.lax.cond(
            state.controlled_player_id != PlayerID.NONE,
            render_arrow_body,
            lambda r: r,
            raster
        )
        
        # --- Render Ball if in flight ---
        ball_in_flight = (state.ball.holder == PlayerID.NONE)
        ball_mask = self.SHAPE_MASKS['ball']

        def render_ball_body(current_raster):
            ball_x = jnp.round(state.ball.x).astype(jnp.int32)
            ball_y = jnp.round(state.ball.y).astype(jnp.int32)
            return self.jr.render_at(current_raster, ball_x, ball_y, ball_mask)

        raster = jax.lax.cond(
            ball_in_flight,
            render_ball_body,
            lambda r: r,
            raster
        )

        # --- Render Scores ---
        player_score_digits = self.jr.int_to_digits(state.player_score, 2)
        enemy_score_digits = self.jr.int_to_digits(state.enemy_score, 2)
        player_score_x = 65
        enemy_score_x = player_score_x+24 
        score_y = 10

        raster = self.jr.render_label_selective(raster, player_score_x, score_y, player_score_digits, self.SHAPE_MASKS['score'], 0, 2, spacing=4)
        raster = self.jr.render_label_selective(raster, enemy_score_x, score_y, enemy_score_digits, self.SHAPE_MASKS['score'], 0, 2, spacing=4)

        # First, always convert the base raster to an image
        final_image = self.jr.render_from_palette(raster, self.PALETTE)

        def apply_play_selection_overlay(image):
            # 1. Apply shadow to the whole image
            shadow_color = jnp.array([0, 0, 0], dtype=jnp.uint8) # Black
            opacity = 0.25
            shadowed_image = (image * (1 - opacity) + shadow_color * opacity).astype(jnp.uint8)

            # 2. Stamp the text on top of the shadowed image
            play_selection_mask = self.SHAPE_MASKS['play_selection']
            
            # Convert text mask to RGB
            text_sprite_rgb = self.PALETTE[play_selection_mask]
            
            # Create alpha mask for the text
            text_alpha_mask = (play_selection_mask != self.jr.TRANSPARENT_ID)[..., None]

            # Position the text
            text_x = (self.consts.WINDOW_WIDTH - play_selection_mask.shape[1]) // 2
            text_y = (self.consts.WINDOW_HEIGHT - play_selection_mask.shape[0]) // 2 - 15

            # Get the slice from the shadowed image
            image_slice = jax.lax.dynamic_slice(
                shadowed_image,
                (text_y, text_x, 0),
                (play_selection_mask.shape[0], play_selection_mask.shape[1], 3)
            )

            # Blend the text onto the slice
            combined_slice = jnp.where(text_alpha_mask, text_sprite_rgb, image_slice)

            # Update the image with the blended slice
            return jax.lax.dynamic_update_slice(
                shadowed_image,
                combined_slice,
                (text_y, text_x, 0)
            )

        def apply_penalty_overlay(image, mask_name):
            # Stamp the penalty text at the bottom
            mask = self.SHAPE_MASKS[mask_name]
            
            # Convert text mask to RGB
            text_sprite_rgb = self.PALETTE[mask]
            
            # Create alpha mask for the text
            text_alpha_mask = (mask != self.jr.TRANSPARENT_ID)[..., None]

            # Position the text at the bottom
            text_x = (self.consts.WINDOW_WIDTH - mask.shape[1]) // 2
            text_y = self.consts.WINDOW_HEIGHT - mask.shape[0] - 25

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

        final_image = jax.lax.cond(
            state.game_mode == GameMode.PLAY_SELECTION,
            apply_play_selection_overlay,
            lambda x: x, 
            final_image
        )

        final_image = jax.lax.cond(
            state.game_mode == GameMode.TRAVEL_PENALTY,
            lambda x: apply_penalty_overlay(x, 'travel'),
            lambda x: x, 
            final_image
        )

        final_image = jax.lax.cond(
            state.game_mode == GameMode.OUT_OF_BOUNDS_PENALTY,
            lambda x: apply_penalty_overlay(x, 'out_of_bounds'),
            lambda x: x, 
            final_image
        )

        return jax.lax.cond(
            state.game_mode == GameMode.CLEARANCE_PENALTY,
            lambda x: apply_penalty_overlay(x, 'clearance'),
            lambda x: x, 
            final_image
        )