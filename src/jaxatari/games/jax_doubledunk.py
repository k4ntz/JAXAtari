from jax import numpy as jnp
from typing import Tuple
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

class Strategy(IntEnum):
    PASS = 0
    JUMP = 1
    SHOOT = 2

PICK_AND_ROLL = jnp.array([Strategy.PASS, Strategy.JUMP, Strategy.SHOOT, Strategy.SHOOT])
GIVE_AND_GO = jnp.array([Strategy.PASS, Strategy.PASS, Strategy.JUMP, Strategy.SHOOT])
MR_INSIDE_SHOOTS = jnp.array([Strategy.JUMP, Strategy.SHOOT, Strategy.SHOOT, Strategy.SHOOT])

@chex.dataclass(frozen=True)
class DunkConstants:
    """Holds all static values for the game like screen dimensions, player speeds, colors, etc."""
    WINDOW_WIDTH: int = 250
    WINDOW_HEIGHT: int = 150
    BALL_SIZE: Tuple[int, int] = (3,3)
    BALL_START: Tuple [int, int] = (122, 100)
    JUMP_STRENGTH: int = 5
    PLAYER_MAX_SPEED: int = 2
    PLAYER_Y_MIN: int = 20
    PLAYER_Y_MAX: int = 120
    PLAYER_X_MIN: int  = 0
    PLAYER_X_MAX: int = 250
    BASKET_POSITION: Tuple[int,int] = (125,10)
    GRAVITY: int = 1
    AREA_3_POINT: Tuple[int,int,int] = (40, 210, 81) # (x_min, x_max, y_arc_connect) - needs a proper function to check if a point is in the 3-point area
    MATCH_STEPS: int = 1200  # number of steps per match (tunable)
    MAX_SCORE: int = 10
    DUNK_RADIUS: int = 18
    BLOCK_RADIUS: int = 14

@chex.dataclass(frozen=True)
class PlayerState:
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    z: chex.Array
    vel_z: chex.Array
    role: chex.Array # can be 0 for defense, 1 for offense
    animation_frame: chex.Array
    animation_direction: chex.Array

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
    shooter_id: chex.Array
    receiver_id: chex.Array
    shooter_pos_x: chex.Array
    shooter_pos_y: chex.Array

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
    strategy: Strategy
    p1_strategy_step: chex.Array           # Tracks progress in p1 strategy (e.g., 1st, 2nd, 3rd button press)
    p2_strategy_step: chex.Array           # Tracks progress in p2 strategy (e.g., 1st, 2nd, 3rd button press)
    controlled_player_id: chex.Array
    key: chex.PRNGKey

@chex.dataclass(frozen=True)
class EntityPosition:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

@chex.dataclass(frozen=True)
class DunkObservation:
    player: EntityPosition
    enemy: EntityPosition
    ball: EntityPosition
    score_player: jnp.ndarray
    score_enemy: jnp.ndarray

@chex.dataclass(frozen=True)
class DunkInfo:
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
            x=jnp.array(state.player1_inside.x),
            y=jnp.array(state.player1_inside.y),
            width=jnp.array(10),  
            height=jnp.array(30), 
        )
        enemy = EntityPosition(
            x=jnp.array(state.player2_inside.x),
            y=jnp.array(state.player2_inside.y),
            width=jnp.array(10),  
            height=jnp.array(30), 
        )
        ball = EntityPosition(
            x=jnp.array(state.ball.x),
            y=jnp.array(state.ball.y),
            width=jnp.array(self.constants.BALL_SIZE[0]),
            height=jnp.array(self.constants.BALL_SIZE[1]),
        )
        return DunkObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            score_player=state.player_score,
            score_enemy=state.enemy_score,
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
    
    def _init_state(self, key) -> DunkGameState:
        """Creates the very first state of the game."""
        return DunkGameState(
            player1_inside=PlayerState(x=125, y=60, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player1_outside=PlayerState(x=80, y=110, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player2_inside=PlayerState(x=170, y=100, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player2_outside=PlayerState(x=50, y=60, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            # Start with a jump ball in the center: no holder and ball sits at the start position
            ball=BallState(x=float(self.constants.BALL_START[0]), y=float(self.constants.BALL_START[1]), vel_x=0.0, vel_y=0.0, holder=PlayerID.NONE, target_x=0.0, target_y=0.0, landing_y=0.0, is_goal=False, shooter_id=PlayerID.NONE, receiver_id=PlayerID.NONE, shooter_pos_x=0, shooter_pos_y=0),
            player_score=0,
            enemy_score=0,
            step_counter=0,
            acceleration_counter=0,
            game_mode=GameMode.PLAY_SELECTION,
            strategy = GIVE_AND_GO,
            p1_strategy_step=0,
            p2_strategy_step=0,
            controlled_player_id=PlayerID.PLAYER1_INSIDE,
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
        new_x = jax.lax.clamp(constants.PLAYER_X_MIN, player.x + vel_x, constants.PLAYER_X_MAX)
        new_y = jax.lax.clamp(constants.PLAYER_Y_MIN, player.y + vel_y, constants.PLAYER_Y_MAX)
        return player.replace(x=new_x, y=new_y, vel_x=vel_x, vel_y=vel_y)

    def _update_players_xy(self, state: DunkGameState, actions: Tuple[int, ...]) -> DunkGameState:
        """Updates the XY positions for all players."""
        p1_inside_action, p1_outside_action, p2_inside_action, p2_outside_action = actions

        updated_p1_inside = self._update_player_xy(state.player1_inside, p1_inside_action, self.constants)
        updated_p1_outside = self._update_player_xy(state.player1_outside, p1_outside_action, self.constants)
        updated_p2_inside = self._update_player_xy(state.player2_inside, p2_inside_action, self.constants)
        updated_p2_outside = self._update_player_xy(state.player2_outside, p2_outside_action, self.constants)

        return state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )

    def _update_player_z(self, player: PlayerState, constants: DunkConstants) -> PlayerState:
        """Applies Z-axis physics (jumping and gravity) to a player."""
        new_z = player.z + player.vel_z
        new_vel_z = player.vel_z - constants.GRAVITY
        has_landed = new_z <= 0
        new_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_z)
        new_vel_z = jax.lax.select(has_landed, jnp.array(0, dtype=jnp.int32), new_vel_z)
        return player.replace(z=new_z, vel_z=new_vel_z)

    def _update_players_z(self, state: DunkGameState) -> DunkGameState:
        """Applies Z-axis physics for all players."""
        updated_p1_inside = self._update_player_z(state.player1_inside, self.constants)
        updated_p1_outside = self._update_player_z(state.player1_outside, self.constants)
        updated_p2_inside = self._update_player_z(state.player2_inside, self.constants)
        updated_p2_outside = self._update_player_z(state.player2_outside, self.constants)
        return state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )

    def _update_players_animations(self, state: DunkGameState) -> DunkGameState:
        """Updates animations for all players."""
        updated_p1_inside = self._update_player_animation(state.player1_inside, (state.ball.holder == PlayerID.PLAYER1_INSIDE))
        updated_p1_outside = self._update_player_animation(state.player1_outside, (state.ball.holder == PlayerID.PLAYER1_OUTSIDE))
        updated_p2_inside = self._update_player_animation(state.player2_inside, (state.ball.holder == PlayerID.PLAYER2_INSIDE))
        updated_p2_outside = self._update_player_animation(state.player2_outside, (state.ball.holder == PlayerID.PLAYER2_OUTSIDE))

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
            shooter_id=PlayerID.NONE, # Reset shooter
            receiver_id=PlayerID.NONE # Reset receiver
        )
        return state.replace(ball=new_ball)

    def _handle_player_actions(self, state: DunkGameState, action: int, key: chex.PRNGKey) -> Tuple[Tuple[int, ...], chex.PRNGKey]:
        """Determines the action for each player based on control state and AI."""
        is_p1_inside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_INSIDE)
        is_p1_outside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_OUTSIDE)
        p1_inside_action = jax.lax.select(is_p1_inside_controlled, action, Action.NOOP)
        p1_outside_action = jax.lax.select(is_p1_outside_controlled, action, Action.NOOP)

        # --- AI Logic ---
        key, p2_action_key, teammate_action_key = random.split(key, 3)
        # We need 4 keys for enemies: 2 for probability checks and 2 for random move selection
        p2_inside_prob_key, p2_outside_prob_key, p2_inside_move_key, p2_outside_move_key = random.split(p2_action_key, 4)
        movement_actions = jnp.array([
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.UPLEFT, Action.UPRIGHT, Action.DOWNLEFT, Action.DOWNRIGHT
        ])

        # --- Teammate AI ---
        p1_inside_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE)
        p1_outside_has_ball = (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        is_p1_inside_teammate_no_ball = ~is_p1_inside_controlled & ~p1_inside_has_ball
        is_p1_outside_teammate_no_ball = ~is_p1_outside_controlled & ~p1_outside_has_ball

        random_teammate_move_idx = random.randint(teammate_action_key, shape=(), minval=0, maxval=8)
        random_teammate_move_action = movement_actions[random_teammate_move_idx]

        p1_inside_action = jax.lax.select(
            is_p1_inside_teammate_no_ball,
            random_teammate_move_action,
            p1_inside_action
        )
        p1_outside_action = jax.lax.select(
            is_p1_outside_teammate_no_ball,
            random_teammate_move_action,
            p1_outside_action
        )

        # --- P2 Inside Logic ---
        p2_inside_has_ball = (state.ball.holder == PlayerID.PLAYER2_INSIDE)
        rand_inside = random.uniform(p2_inside_prob_key)

        # Select a random movement action
        random_inside_move_idx = random.randint(p2_inside_move_key, shape=(), minval=0, maxval=8)
        random_inside_move_action = movement_actions[random_inside_move_idx]

        # Nested select for the 3 outcomes based on probability
        action_if_ball_inside = jax.lax.select(
            rand_inside < 0.05, 
            Action.FIRE,
            random_inside_move_action  
        )
        p2_inside_action = jax.lax.select(p2_inside_has_ball, action_if_ball_inside, random_inside_move_action)

        # --- P2 Outside Logic ---
        p2_outside_has_ball = (state.ball.holder == PlayerID.PLAYER2_OUTSIDE)
        rand_outside = random.uniform(p2_outside_prob_key)

        # Select a random movement action
        random_outside_move_idx = random.randint(p2_outside_move_key, shape=(), minval=0, maxval=8)
        random_outside_move_action = movement_actions[random_outside_move_idx]

        # Nested select for the 3 outcomes based on probability
        action_if_ball_outside = jax.lax.select(
            rand_outside < 0.05, 
            Action.FIRE,
            random_outside_move_action  
        )
        p2_outside_action = jax.lax.select(p2_outside_has_ball, action_if_ball_outside, random_outside_move_action)
        
        actions = (p1_inside_action, p1_outside_action, p2_inside_action, p2_outside_action)
        return actions, key

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

    def _handle_passing(self, state: DunkGameState, actions: Tuple[int, ...]) -> BallState:
        """Handles the logic for passing the ball."""
        p1_inside_action, p1_outside_action, p2_inside_action, p2_outside_action = actions
        ball_state = state.ball
        is_pass_step = (state.strategy[state.p1_strategy_step] == Strategy.PASS)

        is_p1_inside_passing = is_pass_step & (ball_state.holder == PlayerID.PLAYER1_INSIDE) & jnp.any(jnp.asarray(p1_inside_action) == jnp.asarray(list(_PASS_ACTIONS)))
        is_p1_outside_passing = is_pass_step & (ball_state.holder == PlayerID.PLAYER1_OUTSIDE) & jnp.any(jnp.asarray(p1_outside_action) == jnp.asarray(list(_PASS_ACTIONS)))
        is_p2_inside_passing = (ball_state.holder == PlayerID.PLAYER2_INSIDE) & jnp.any(jnp.asarray(p2_inside_action) == jnp.asarray(list(_PASS_ACTIONS)))
        is_p2_outside_passing = (ball_state.holder == PlayerID.PLAYER2_OUTSIDE) & jnp.any(jnp.asarray(p2_outside_action) == jnp.asarray(list(_PASS_ACTIONS)))
        is_passing = is_p1_inside_passing | is_p1_outside_passing | is_p2_inside_passing | is_p2_outside_passing

        receiver_id = jax.lax.select(is_p1_inside_passing, PlayerID.PLAYER1_OUTSIDE,
                      jax.lax.select(is_p1_outside_passing, PlayerID.PLAYER1_INSIDE,
                      jax.lax.select(is_p2_inside_passing, PlayerID.PLAYER2_OUTSIDE,
                      jax.lax.select(is_p2_outside_passing, PlayerID.PLAYER2_INSIDE, PlayerID.NONE))))

        passer_x = jax.lax.select(is_p1_inside_passing, state.player1_inside.x,
                   jax.lax.select(is_p1_outside_passing, state.player1_outside.x,
                   jax.lax.select(is_p2_inside_passing, state.player2_inside.x,
                   jax.lax.select(is_p2_outside_passing, state.player2_outside.x, 0))))
        passer_y = jax.lax.select(is_p1_inside_passing, state.player1_inside.y,
                   jax.lax.select(is_p1_outside_passing, state.player1_outside.y,
                   jax.lax.select(is_p2_inside_passing, state.player2_inside.y,
                   jax.lax.select(is_p2_outside_passing, state.player2_outside.y, 0))))
        receiver_x = jax.lax.select(is_p1_inside_passing, state.player1_outside.x,
                     jax.lax.select(is_p1_outside_passing, state.player1_inside.x,
                     jax.lax.select(is_p2_inside_passing, state.player2_outside.x,
                     jax.lax.select(is_p2_outside_passing, state.player2_inside.x, 0))))
        receiver_y = jax.lax.select(is_p1_inside_passing, state.player1_outside.y,
                     jax.lax.select(is_p1_outside_passing, state.player1_inside.y,
                     jax.lax.select(is_p2_inside_passing, state.player2_outside.y,
                     jax.lax.select(is_p2_outside_passing, state.player2_inside.y, 0))))

        passer_pos = jnp.array([passer_x, passer_y], dtype=jnp.float32)
        receiver_pos = jnp.array([receiver_x, receiver_y], dtype=jnp.float32)
        direction = receiver_pos - passer_pos
        norm = jnp.sqrt(jnp.sum(direction**2))
        safe_norm = jnp.where(norm == 0, 1.0, norm)
        ball_speed = 8.0
        vel = (direction / safe_norm) * ball_speed

        new_ball_state = jax.lax.cond(
            is_passing,
            lambda b: b.replace(x=passer_x.astype(jnp.float32), y=passer_y.astype(jnp.float32), vel_x=vel[0], vel_y=vel[1], holder=PlayerID.NONE, receiver_id=receiver_id),
            lambda b: b,
            ball_state
        )

        is_p1_passing = is_p1_inside_passing | is_p1_outside_passing
        step_increment = jax.lax.select(is_p1_passing, 1, 0)
        return new_ball_state, step_increment

    def _handle_shooting(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[BallState, chex.PRNGKey]:
        """Handles the logic for shooting the ball."""
        p1_inside_action, p1_outside_action, p2_inside_action, p2_outside_action = actions
        ball_state = state.ball
        is_shoot_step = (state.strategy[state.p1_strategy_step] == Strategy.SHOOT)

        is_p1_inside_shooting = is_shoot_step & (ball_state.holder == PlayerID.PLAYER1_INSIDE) & jnp.any(jnp.asarray(p1_inside_action) == jnp.asarray(list(_SHOOT_ACTIONS)))
        is_p1_outside_shooting = is_shoot_step &(ball_state.holder == PlayerID.PLAYER1_OUTSIDE) & jnp.any(jnp.asarray(p1_outside_action) == jnp.asarray(list(_SHOOT_ACTIONS)))
        is_p2_inside_shooting = (ball_state.holder == PlayerID.PLAYER2_INSIDE) & jnp.any(jnp.asarray(p2_inside_action) == jnp.asarray(list(_SHOOT_ACTIONS)))
        is_p2_outside_shooting = (ball_state.holder == PlayerID.PLAYER2_OUTSIDE) & jnp.any(jnp.asarray(p2_outside_action) == jnp.asarray(list(_SHOOT_ACTIONS)))
        is_shooting = is_p1_inside_shooting | is_p1_outside_shooting | is_p2_inside_shooting | is_p2_outside_shooting

        shooter_id = jax.lax.select(is_p1_inside_shooting, PlayerID.PLAYER1_INSIDE,
                     jax.lax.select(is_p1_outside_shooting, PlayerID.PLAYER1_OUTSIDE,
                     jax.lax.select(is_p2_inside_shooting, PlayerID.PLAYER2_INSIDE,
                     jax.lax.select(is_p2_outside_shooting, PlayerID.PLAYER2_OUTSIDE, PlayerID.NONE))))
        shooter_x = jax.lax.select(is_p1_inside_shooting, state.player1_inside.x,
                    jax.lax.select(is_p1_outside_shooting, state.player1_outside.x,
                    jax.lax.select(is_p2_inside_shooting, state.player2_inside.x,
                    jax.lax.select(is_p2_outside_shooting, state.player2_outside.x, 0))))
        shooter_y = jax.lax.select(is_p1_inside_shooting, state.player1_inside.y,
                    jax.lax.select(is_p1_outside_shooting, state.player1_outside.y,
                    jax.lax.select(is_p2_inside_shooting, state.player2_inside.y,
                    jax.lax.select(is_p2_outside_shooting, state.player2_outside.y, 0))))

        key, offset_key_x = random.split(key)
        offset_x = random.uniform(offset_key_x, shape=(), minval=-10, maxval=10)
        is_goal = (offset_x >= -5) & (offset_x <= 5)

        basket_pos = jnp.array([self.constants.BASKET_POSITION[0], self.constants.BASKET_POSITION[1]], dtype=jnp.float32)
        target_pos = basket_pos + jnp.array([offset_x, 0.0])
        shooter_pos = jnp.array([shooter_x, shooter_y], dtype=jnp.float32)
        # shooter z (height) for dunk/block checks
        shooter_z = jax.lax.select(is_p1_inside_shooting, state.player1_inside.z,
                  jax.lax.select(is_p1_outside_shooting, state.player1_outside.z,
                  jax.lax.select(is_p2_inside_shooting, state.player2_inside.z,
                  jax.lax.select(is_p2_outside_shooting, state.player2_outside.z, 0))))
        shoot_direction = target_pos - shooter_pos
        shoot_norm = jnp.sqrt(jnp.sum(shoot_direction**2))
        shoot_safe_norm = jnp.where(shoot_norm == 0, 1.0, shoot_norm)
        shoot_speed = 8.0
        shoot_vel = (shoot_direction / shoot_safe_norm) * shoot_speed

        # Basic blocking: if an opponent jumps near the shooter during the shot, they can block it
        # Check opponents (for P1 shots, opponents are P2 players; vice-versa)
        def check_blocking():
            # opponent positions and heights
            opp1_x = state.player2_inside.x
            opp1_y = state.player2_inside.y
            opp1_z = state.player2_inside.z
            opp2_x = state.player2_outside.x
            opp2_y = state.player2_outside.y
            opp2_z = state.player2_outside.z

            d1 = jnp.sqrt((opp1_x - shooter_x)**2 + (opp1_y - shooter_y)**2)
            d2 = jnp.sqrt((opp2_x - shooter_x)**2 + (opp2_y - shooter_y)**2)

            can_block1 = (opp1_z > 0) & (d1 < self.constants.BLOCK_RADIUS)
            can_block2 = (opp2_z > 0) & (d2 < self.constants.BLOCK_RADIUS)

            blocked_by = jax.lax.select(can_block1, PlayerID.PLAYER2_INSIDE, PlayerID.NONE)
            blocked_by = jax.lax.select(can_block2, PlayerID.PLAYER2_OUTSIDE, blocked_by)
            return blocked_by

        blocked_by = jax.lax.cond(
            is_shooting & ((shooter_id == PlayerID.PLAYER1_INSIDE) | (shooter_id == PlayerID.PLAYER1_OUTSIDE)),
            lambda: check_blocking(),
            lambda: PlayerID.NONE
        )

        # Determine whether this shot should be a dunk (inside player jumping near basket)
        dist_to_basket = jnp.sqrt(jnp.sum((shooter_pos - basket_pos)**2))
        is_dunk = (dist_to_basket < self.constants.DUNK_RADIUS) & (shooter_z > 0)

        def make_shot(b):
            # If blocked by opponent, possession goes to blocker
            b = b.replace(x=shooter_x.astype(jnp.float32), y=shooter_y.astype(jnp.float32), vel_x=shoot_vel[0], vel_y=shoot_vel[1], holder=PlayerID.NONE, target_x=target_pos[0], target_y=target_pos[1], is_goal=is_goal, shooter_id=shooter_id, receiver_id=PlayerID.NONE, shooter_pos_x=shooter_x.astype(jnp.int32), shooter_pos_y=shooter_y.astype(jnp.int32))
            b = jax.lax.cond(blocked_by != PlayerID.NONE, lambda bb: bb.replace(holder=blocked_by, vel_x=0.0, vel_y=0.0, is_goal=False, shooter_id=PlayerID.NONE), lambda bb: bb, b)
            # If dunk, bump is_goal to True and make target the rim
            b = jax.lax.cond(is_dunk, lambda bb: bb.replace(is_goal=True, target_x=basket_pos[0], target_y=basket_pos[1]), lambda bb: bb, b)
            return b

        new_ball_state = jax.lax.cond(
            is_shooting,
            make_shot,
            lambda b: b,
            ball_state
        )
        is_p1_shooting = is_p1_inside_shooting | is_p1_outside_shooting
        step_increment = jax.lax.select(is_p1_shooting, 1, 0)
        return new_ball_state, key, step_increment

    def _handle_stealing(self, state: DunkGameState, actions: Tuple[int, ...]) -> BallState:
        """Handles the logic for stealing the ball."""
        human_action = actions[0] # The primary action from the user
        ball_state = state.ball

        steal_radius = 5.0
        stealer_id = state.controlled_player_id
        stealer_x = jax.lax.select(stealer_id == PlayerID.PLAYER1_INSIDE, state.player1_inside.x, state.player1_outside.x)
        stealer_y = jax.lax.select(stealer_id == PlayerID.PLAYER1_INSIDE, state.player1_inside.y, state.player1_outside.y)

        is_trying_to_steal = jnp.any(jnp.asarray(human_action) == jnp.asarray(list(STEAL_ACTIONS)))
        is_close_to_ball = jnp.sqrt((stealer_x - ball_state.x)**2 + (stealer_y - ball_state.y)**2) < steal_radius
        is_p1_team_holder = (ball_state.holder == PlayerID.PLAYER1_INSIDE) | (ball_state.holder == PlayerID.PLAYER1_OUTSIDE)
        can_steal_from_holder = ~is_p1_team_holder
        is_stealing = is_trying_to_steal & is_close_to_ball & can_steal_from_holder

        new_ball_state = jax.lax.cond(
            is_stealing,
            lambda b: b.replace(holder=stealer_id, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            ball_state
        )
        return new_ball_state

    def _handle_jump(self, state: DunkGameState, player: PlayerState, action: int, constants: DunkConstants) -> chex.Array:
        """Calculates the vertical impulse for a jump."""
        is_jump_step = (state.strategy[state.p1_strategy_step] == Strategy.JUMP)
        can_jump = is_jump_step & jnp.any(jnp.asarray(action) == jnp.asarray(list(_JUMP_ACTIONS)))
        vel_z = jax.lax.select(can_jump, constants.JUMP_STRENGTH, jnp.array(0, dtype=jnp.int32))
        new_vel_z = jax.lax.select(vel_z > 0, vel_z, player.vel_z)
        did_jump = jax.lax.select(can_jump, 1, 0)
        return player.replace(vel_z=new_vel_z), did_jump

    def _handle_offense_actions(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[DunkGameState, chex.PRNGKey]:
        """Handles offensive actions: passing and shooting."""
        # Passing
        ball_state_after_pass, pass_inc = self._handle_passing(state, actions)
        state = state.replace(ball=ball_state_after_pass)

        # Shooting
        ball_state_after_shot, key, shot_inc = self._handle_shooting(state, actions, key)
        state = state.replace(ball=ball_state_after_shot)

        step_increment = pass_inc + shot_inc
        return state, key, step_increment

    def _handle_defense_actions(self, state: DunkGameState, actions: Tuple[int, ...]) -> DunkGameState:
        """Handles defensive actions: stealing."""
        ball_state_after_steal = self._handle_stealing(state, actions)
        state = state.replace(ball=ball_state_after_steal)
        return state

    def _handle_interactions(self, state: DunkGameState, actions: Tuple[int, ...], key: chex.PRNGKey) -> Tuple[DunkGameState, chex.PRNGKey]:
        """Handles all player interactions: jump, pass, shoot, steal."""
        p1_inside_action, p1_outside_action, p2_inside_action, p2_outside_action = actions

        updated_p1_inside, p1_in_jumped = self._handle_jump(state, state.player1_inside, p1_inside_action, self.constants)
        updated_p1_outside, p1_out_jumped = self._handle_jump(state, state.player1_outside, p1_outside_action, self.constants)
        updated_p2_inside, p2_in_jumped = self._handle_jump(state, state.player2_inside, p2_inside_action, self.constants)
        updated_p2_outside, p2_out_jumped = self._handle_jump(state, state.player2_outside, p2_outside_action, self.constants)

        state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
        )

        # 2. Handle ball actions
        state, key, offense_increment = self._handle_offense_actions(state, actions, key)
        state = self._handle_defense_actions(state, actions)

        # 3. Calculate total p1_strategy_step increment
        total_increment = p1_in_jumped + p1_out_jumped + offense_increment
        
        # 4. Update p1_strategy_step
        new_p1_strategy_step = jnp.minimum(state.p1_strategy_step + total_increment, len(state.strategy)-1)
        
        # 5. Print if changed
        # We use jax.lax.cond to ensure we only print when an action actually occurred
        jax.lax.cond(
            total_increment > 0,
            lambda x: jax.debug.print("Play Step: {}", x),
            lambda x: None,
            new_p1_strategy_step
        )

        return state.replace(p1_strategy_step=new_p1_strategy_step), key

    def _update_ball(self, state: DunkGameState) -> DunkGameState:
        """Handles ball movement, goals, misses, catches, and possession changes."""
        ball_in_flight = (state.ball.holder == PlayerID.NONE)
        dist_to_target = jnp.sqrt((state.ball.x - state.ball.target_x)**2 + (state.ball.y - state.ball.target_y)**2)
        is_already_falling = (state.ball.vel_x == 0) & jnp.isclose(state.ball.vel_y, 2.0)
        reached_target = ball_in_flight & (dist_to_target < 5.0) & ~is_already_falling
        is_goal_scored = reached_target & state.ball.is_goal

        def on_goal(s):
            key, reset_key = random.split(s.key)
            is_p1_scorer = (s.ball.shooter_id == PlayerID.PLAYER1_INSIDE) | (s.ball.shooter_id == PlayerID.PLAYER1_OUTSIDE)

            shooter_x = s.ball.shooter_pos_x
            shooter_y = s.ball.shooter_pos_y

            x_min, x_max, y_arc = self.constants.AREA_3_POINT
            is_3_pointer = (shooter_y > y_arc) & (shooter_x > x_min) & (shooter_x < x_max)

            points = jax.lax.select(is_3_pointer, 3, 2)

            new_player_score = s.player_score + points * is_p1_scorer
            new_enemy_score = s.enemy_score + points * (1 - is_p1_scorer)

            new_state = self._init_state(reset_key).replace(player_score=new_player_score, enemy_score=new_enemy_score, step_counter=s.step_counter)
            new_ball_holder = jax.lax.select(is_p1_scorer, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER1_INSIDE)
            return new_state.replace(ball=new_state.ball.replace(holder=new_ball_holder))

        def continue_play(s):
            # Handle miss
            is_miss = reached_target & ~s.ball.is_goal
            s = jax.lax.cond(is_miss, self._handle_miss, lambda s_: s_, s)
            b_state = s.ball

            # --- New logic for passing ---
            is_passing = b_state.receiver_id != PlayerID.NONE

            def update_pass_trajectory(b):
                receiver_x = jax.lax.switch(
                    b.receiver_id - 1,
                    [
                        lambda: s.player1_inside.x,
                        lambda: s.player1_outside.x,
                        lambda: s.player2_inside.x,
                        lambda: s.player2_outside.x,
                    ],
                )
                receiver_y = jax.lax.switch(
                    b.receiver_id - 1,
                    [
                        lambda: s.player1_inside.y,
                        lambda: s.player1_outside.y,
                        lambda: s.player2_inside.y,
                        lambda: s.player2_outside.y,
                    ],
                )

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
            players = [s.player1_inside, s.player1_outside, s.player2_inside, s.player2_outside]
            player_ids = [PlayerID.PLAYER1_INSIDE, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER2_INSIDE, PlayerID.PLAYER2_OUTSIDE]
            
            catch_radius_sq = 49.0 # optimal value: tried different values
            ball_in_flight_after_physics = (b_state.holder == PlayerID.NONE)

            def check_catch(p, pid, current_ball_state):
                dist_sq = (current_ball_state.x - p.x)**2 + (current_ball_state.y - p.y)**2
                is_caught = ball_in_flight_after_physics & (dist_sq < catch_radius_sq)
                return jax.lax.cond(is_caught, lambda b: b.replace(holder=pid, vel_x=0.0, vel_y=0.0, receiver_id=PlayerID.NONE), lambda b: b, current_ball_state)

            # This loop is symbolic; JAX will unroll it.
            for p, pid in zip(players, player_ids):
                 b_state = check_catch(p, pid, b_state)

            # Update ball position if held
            def update_ball_pos_if_held(b, p, pid):
                return jax.lax.cond(b.holder == pid, lambda: b.replace(x=p.x.astype(jnp.float32), y=p.y.astype(jnp.float32)), lambda: b)
            
            b_state = update_ball_pos_if_held(b_state, s.player1_inside, PlayerID.PLAYER1_INSIDE)
            b_state = update_ball_pos_if_held(b_state, s.player1_outside, PlayerID.PLAYER1_OUTSIDE)
            b_state = update_ball_pos_if_held(b_state, s.player2_inside, PlayerID.PLAYER2_INSIDE)
            b_state = update_ball_pos_if_held(b_state, s.player2_outside, PlayerID.PLAYER2_OUTSIDE)

            # Update controlled player
            is_p1_team_holder = (b_state.holder == PlayerID.PLAYER1_INSIDE) | (b_state.holder == PlayerID.PLAYER1_OUTSIDE)
            new_controlled_player_id = jax.lax.select(is_p1_team_holder, b_state.holder, s.controlled_player_id)
            
            return s.replace(ball=b_state, controlled_player_id=new_controlled_player_id)

        final_state = jax.lax.cond(is_goal_scored, on_goal, continue_play, state)
        return final_state

    def _handle_play_selection(self, state: DunkGameState, action: int) -> DunkGameState:
        """Handles the play selection mode."""

        def select_strategy(strategy):
            # When a strategy is selected, we reset the game to its initial state
            # but keep the selected strategy and switch to IN_PLAY mode.
            # We also keep the current key.
            new_state = self._init_state(state.key)
            return new_state.replace(
                strategy=strategy,
                game_mode=GameMode.IN_PLAY,
            )

        # Check for each strategy selection action
        state = jax.lax.cond(
            action == Action.UPFIRE,
            lambda s: select_strategy(PICK_AND_ROLL),
            lambda s: s,
            state
        )
        state = jax.lax.cond(
            action == Action.DOWNFIRE,
            lambda s: select_strategy(GIVE_AND_GO),
            lambda s: s,
            state
        )
        state = jax.lax.cond(
            action == Action.RIGHTFIRE,
            lambda s: select_strategy(MR_INSIDE_SHOOTS),
            lambda s: s,
            state
        )
        return state

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

        def run_game_step(s):
            # 1. Determine actions for all players
            actions, key = self._handle_player_actions(s, action, s.key)

            # 2. Update player XY movement
            s = self._update_players_xy(s, actions)

            # 3. Handle interactions (jump, pass, shoot, steal)
            s, key = self._handle_interactions(s, actions, key)
            
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
        is_max_steps_reached = state.step_counter >= self.constants.MATCH_STEPS
        is_max_score_reached = (state.player_score >= self.constants.MAX_SCORE) | (state.enemy_score >= self.constants.MAX_SCORE)
        return is_max_steps_reached | is_max_score_reached

    def _get_info(self, state: DunkGameState) -> DunkInfo:
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
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'player', 'type': 'group', 'files': [f'player_{i}.npy' for i in range(10)]},
            {'name': 'player_no_ball', 'type': 'single', 'file': 'player_no_ball.npy'},
            {'name': 'enemy', 'type': 'group', 'files': [f'enemy_{i}.npy' for i in range(10)]},
            {'name': 'enemy_no_ball', 'type': 'single', 'file': 'enemy_no_ball.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
            {'name': 'player_arrow', 'type': 'single', 'file': 'player_arrow.npy'},
            {'name': 'score', 'type': 'digits', 'pattern': 'score_{}.npy', 'files': [f'score_{i}.npy' for i in range(21)]},
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

        # Identify team and ball holder for each player
        is_team1 = jnp.array([True, True, False, False])
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
            player_idx = sort_indices[i]

            x = all_players_x[player_idx]
            visual_y = visual_ys[player_idx]
            anim_frame = all_players_anim_frame[player_idx]
            is_p1 = is_team1[player_idx]
            p_has_ball = has_ball[player_idx]

            # Select the correct mask based on team, ball possession, and animation frame
            player_mask_with_ball = self.SHAPE_MASKS['player'][anim_frame]
            player_mask_no_ball = self.SHAPE_MASKS['player_no_ball']
            enemy_mask_with_ball = self.SHAPE_MASKS['enemy'][anim_frame]
            enemy_mask_no_ball = self.SHAPE_MASKS['enemy_no_ball']

            mask_with_ball = jax.lax.select(is_p1, player_mask_with_ball, enemy_mask_with_ball)
            mask_no_ball = jax.lax.select(is_p1, player_mask_no_ball, enemy_mask_no_ball)

            final_mask = jax.lax.select(p_has_ball, mask_with_ball, mask_no_ball)

            return self.jr.render_at(current_raster, x, visual_y, final_mask)

        raster = jax.lax.fori_loop(0, 4, render_player_body, raster)

        # --- Render Controlled Player Arrow ---        
        # Only render if a player is controlled (ID is not NONE)
        def render_arrow_body(current_raster):
            controlled_x = all_players_x[state.controlled_player_id - 1]
            controlled_visual_y = visual_ys[state.controlled_player_id - 1]
            arrow_mask = self.SHAPE_MASKS['player_arrow']
            arrow_height = arrow_mask.shape[0]
            
            # Position the arrow above the player
            arrow_y = controlled_visual_y - arrow_height
            
            # Center the arrow horizontally over the player
            player_width = 10 # from placeholder
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
        player_score_x = 110
        enemy_score_x = 135 # Start of player score + width of player score + gap
        score_y = 10

        raster = self.jr.render_label_selective(raster, player_score_x, score_y, player_score_digits, self.SHAPE_MASKS['score'], 0, 2, spacing=4)
        raster = self.jr.render_label_selective(raster, enemy_score_x, score_y, enemy_score_digits, self.SHAPE_MASKS['score'], 0, 2, spacing=4)

        return self.jr.render_from_palette(raster, self.PALETTE)