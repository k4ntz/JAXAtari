from jax import numpy as jnp
from typing import Tuple
import jax.lax
import chex
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from functools import partial
import os
from enum import IntEnum

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

@chex.dataclass(frozen=True)
class DunkConstants:
    """
    Holds all static values for the game like screen dimensions, player speeds, colors, etc.
    """
    WINDOW_WIDTH: int = 250
    WINDOW_HEIGHT: int = 150
    BALL_SIZE: Tuple[int, int] = (3,3)
    BALL_START: Tuple [int, int] = (100, 70)
    JUMP_STRENGTH: int = 5 #adjustable if necessary and more of a placeholder value 
    PLAYER_MAX_SPEED: int = 2 #adjustable if necessary and more of a placeholder value
    PLAYER_Y_MIN: int = 20
    PLAYER_Y_MAX: int = 150
    PLAYER_X_MIN: int  = 0
    PLAYER_X_MAX: int = 250
    PLAYER_ROLES: Tuple[int,int] = (0,1) #0 = Offence, 1 = Defence (might be doable with booleans as well)
    BASKET_POSITION: Tuple[int,int] = (125,10)
    BASKET_BUFFER: int = 3 #this should translate to [BASKET_POSITION[0]-buffer:BASKET_POSITION[0]+buffer] being the valid goal area width-wise
    GRAVITY: int = 1 # Downward acceleration due to gravity
    AREA_3_POINT: Tuple[int,int,int] = (40, 210, 81) # (x_min, x_max, y_arc_connect) - needs a proper function to check if a point is in the 3-point area

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

    # New fields for game logic:
    game_mode: chex.Array           # Current mode (PLAY_SELECTION or IN_PLAY)
    offensive_play: chex.Array      # The selected offensive play
    defensive_play: chex.Array      # The selected defensive play
    play_step: chex.Array           # Tracks progress within a play (e.g., 1st, 2nd, 3rd button press)
    controlled_player_id: chex.Array

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

class DoubleDunk(JaxEnvironment[DunkGameState, DunkObservation, DunkInfo, DunkConstants]):
    
    def __init__(self):
        """
        Initialize the game environment.
        """
        self.constants = DunkConstants()
        self.renderer = DunkRenderer(self.constants)

    def reset(self, key=None) -> Tuple[DunkObservation, DunkGameState]:
        """
        Resets the environment to the initial state.
        """
        state = self._init_state()
        obs = self._get_observation(state)
        return obs, state

    def _get_observation(self, state: DunkGameState) -> DunkObservation:
        """
        Converts the environment state to an observation.
        """
        # For now, we'll treat player1_inside as the main 'player'
        # and player2_inside as the 'enemy' for the observation.
        player = EntityPosition(
            x=jnp.array(state.player1_inside.x),
            y=jnp.array(state.player1_inside.y),
            width=jnp.array(10),  # Placeholder width
            height=jnp.array(30), # Placeholder height
        )
        enemy = EntityPosition(
            x=jnp.array(state.player2_inside.x),
            y=jnp.array(state.player2_inside.y),
            width=jnp.array(10),  # Placeholder width
            height=jnp.array(30), # Placeholder height
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
        """
        Returns the action space of the environment.
        """
        return [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT,
            Action.DOWN, Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT,
            Action.DOWNLEFT, Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE,
            Action.DOWNFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]

    # Define action sets
    _MOVE_LEFT_ACTIONS = {Action.LEFT, Action.UPLEFT, Action.DOWNLEFT}
    _MOVE_RIGHT_ACTIONS = {Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT}
    _MOVE_UP_ACTIONS = {Action.UP, Action.UPLEFT, Action.UPRIGHT}
    _MOVE_DOWN_ACTIONS = {Action.DOWN, Action.DOWNLEFT, Action.DOWNRIGHT}
    _JUMP_ACTIONS = {Action.FIRE} 
    _PASS_ACTIONS = {Action.DOWNFIRE} 

    def observation_space(self):
        """
        Returns the observation space of the environment.
        """
        # This is a placeholder based on Pong. It should be updated
        # with the correct dimensions and types for DoubleDunk.
        return {
            "player": {
                "x": "Box(low=0, high=200, shape=(), dtype=jnp.int32)",
                "y": "Box(low=0, high=240, shape=(), dtype=jnp.int32)",
                "width": "Box(low=0, high=200, shape=(), dtype=jnp.int32)",
                "height": "Box(low=0, high=240, shape=(), dtype=jnp.int32)",
            },
            "enemy": {
                "x": "Box(low=0, high=200, shape=(), dtype=jnp.int32)",
                "y": "Box(low=0, high=240, shape=(), dtype=jnp.int32)",
                "width": "Box(low=0, high=200, shape=(), dtype=jnp.int32)",
                "height": "Box(low=0, high=240, shape=(), dtype=jnp.int32)",
            },
            "ball": {
                "x": "Box(low=0, high=200, shape=(), dtype=jnp.int32)",
                "y": "Box(low=0, high=240, shape=(), dtype=jnp.int32)",
                "width": "Box(low=0, high=200, shape=(), dtype=jnp.int32)",
                "height": "Box(low=0, high=240, shape=(), dtype=jnp.int32)",
            },
            "score_player": "Box(low=0, high=99, shape=(), dtype=jnp.int32)",
            "score_enemy": "Box(low=0, high=99, shape=(), dtype=jnp.int32)",
        }
    
    def _init_state(self) -> DunkGameState:
        """
        Creates the very first state of the game.
        Use values from self.constants.
        """
        return DunkGameState(
            player1_inside=PlayerState(x=125, y=80, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player1_outside=PlayerState(x=80, y=110, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player2_inside=PlayerState(x=170, y=100, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player2_outside=PlayerState(x=50, y=60, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            ball=BallState(x=0.0, y=0.0, vel_x=0.0, vel_y=0.0, holder=PlayerID.PLAYER1_INSIDE),
            player_score=0,
            enemy_score=0,
            step_counter=0,
            acceleration_counter=0,
            game_mode=GameMode.PLAY_SELECTION,
            offensive_play=OffensivePlay.NONE,
            defensive_play=DefensivePlay.NONE,
            play_step=0,
            controlled_player_id=PlayerID.PLAYER1_INSIDE,
        )



    def _get_player_action_effects(self, action: int, player_z: chex.Array, constants: DunkConstants) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Determines the velocity for 8-way movement and the impulse for Z-axis jumps.
        """
        # --- X/Y Movement on the ground plane ---
        action_jnp = jnp.asarray(action)

        is_moving_left = jnp.any(action_jnp == jnp.asarray(list(self._MOVE_LEFT_ACTIONS)))
        is_moving_right = jnp.any(action_jnp == jnp.asarray(list(self._MOVE_RIGHT_ACTIONS)))

        vel_x = jnp.array(0, dtype=jnp.int32)
        vel_x = jax.lax.select(is_moving_left, -constants.PLAYER_MAX_SPEED, vel_x)
        vel_x = jax.lax.select(is_moving_right, constants.PLAYER_MAX_SPEED, vel_x)

        is_moving_up = jnp.any(action_jnp == jnp.asarray(list(self._MOVE_UP_ACTIONS)))
        is_moving_down = jnp.any(action_jnp == jnp.asarray(list(self._MOVE_DOWN_ACTIONS)))

        vel_y = jnp.array(0, dtype=jnp.int32)
        vel_y = jax.lax.select(is_moving_up, -constants.PLAYER_MAX_SPEED, vel_y)
        vel_y = jax.lax.select(is_moving_down, constants.PLAYER_MAX_SPEED, vel_y)

        # --- Z-Axis Jump Impulse ---
        # A jump can only be initiated if the player is on the ground (z=0) and presses FIRE
        can_jump = (player_z == 0) & jnp.any(action_jnp == jnp.asarray(list(self._JUMP_ACTIONS)))
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
        vel_x, vel_y, jump_impulse = self._get_player_action_effects(action, player.z, constants)

        # Set the player's X/Y velocity based on the action
        updated_player = player.replace(vel_x=vel_x, vel_y=vel_y)

        # If there's a jump impulse, apply it. Otherwise, keep the existing vel_z.
        # This prevents the vertical velocity from being reset to 0 every frame.
        new_vel_z = jax.lax.select(
            jump_impulse > 0,
            jump_impulse,
            updated_player.vel_z
        )
        updated_player = updated_player.replace(vel_z=new_vel_z)

        # Apply physics (movement, gravity, collisions) to the player
        updated_player = self._update_player_physics(updated_player, constants)

        return updated_player

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DunkGameState, action: int) -> Tuple[DunkObservation, DunkGameState, float, bool, DunkInfo]:
        """
        Takes an action in the game and returns the new game state.
        """
        # 1. Update player states based on actions
        is_p1_inside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_INSIDE)
        is_p1_outside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_OUTSIDE)

        p1_inside_action = jax.lax.select(is_p1_inside_controlled, action, Action.NOOP)
        p1_outside_action = jax.lax.select(is_p1_outside_controlled, action, Action.NOOP)

        updated_p1_inside = self._update_player(state.player1_inside, p1_inside_action, self.constants)
        updated_p1_outside = self._update_player(state.player1_outside, p1_outside_action, self.constants)
        
        updated_p2_inside = self._update_player(state.player2_inside, Action.NOOP, self.constants)
        updated_p2_outside = self._update_player(state.player2_outside, Action.NOOP, self.constants)

        # 2. Update animation for the player who has the ball
        p1_inside_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE)
        p1_outside_has_ball = (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        p2_inside_has_ball = (state.ball.holder == PlayerID.PLAYER2_INSIDE)
        p2_outside_has_ball = (state.ball.holder == PlayerID.PLAYER2_OUTSIDE)

        # --- Animation for Player 1 Inside ---
        p1_inside_anim_frame = updated_p1_inside.animation_frame
        p1_inside_anim_dir = updated_p1_inside.animation_direction

        # Calculate next frame if it has the ball
        p1_inside_new_dir = jax.lax.cond(p1_inside_anim_frame >= 9, lambda: -1, lambda: p1_inside_anim_dir)
        p1_inside_new_dir = jax.lax.cond(p1_inside_anim_frame <= 0, lambda: 1, lambda: p1_inside_new_dir)
        p1_inside_new_frame = p1_inside_anim_frame + p1_inside_new_dir

        # Update if it has the ball, otherwise reset
        final_p1_inside_frame = jax.lax.select(p1_inside_has_ball, p1_inside_new_frame, 0)
        final_p1_inside_dir = jax.lax.select(p1_inside_has_ball, p1_inside_new_dir, 1) # Reset direction to 1
        
        updated_p1_inside = updated_p1_inside.replace(
            animation_frame=final_p1_inside_frame,
            animation_direction=final_p1_inside_dir
        )

        # --- Animation for Player 1 Outside ---
        p1_outside_anim_frame = updated_p1_outside.animation_frame
        p1_outside_anim_dir = updated_p1_outside.animation_direction

        # Calculate next frame if it has the ball
        p1_outside_new_dir = jax.lax.cond(p1_outside_anim_frame >= 9, lambda: -1, lambda: p1_outside_anim_dir)
        p1_outside_new_dir = jax.lax.cond(p1_outside_anim_frame <= 0, lambda: 1, lambda: p1_outside_new_dir)
        p1_outside_new_frame = p1_outside_anim_frame + p1_outside_new_dir

        # Update if it has the ball, otherwise reset
        final_p1_outside_frame = jax.lax.select(p1_outside_has_ball, p1_outside_new_frame, 0)
        final_p1_outside_dir = jax.lax.select(p1_outside_has_ball, p1_outside_new_dir, 1)

        updated_p1_outside = updated_p1_outside.replace(
            animation_frame=final_p1_outside_frame,
            animation_direction=final_p1_outside_dir
        )

        # --- Animation for Player 2 Inside ---
        p2_inside_anim_frame = updated_p2_inside.animation_frame
        p2_inside_anim_dir = updated_p2_inside.animation_direction

        # Calculate next frame if it has the ball
        p2_inside_new_dir = jax.lax.cond(p2_inside_anim_frame >= 9, lambda: -1, lambda: p2_inside_anim_dir)
        p2_inside_new_dir = jax.lax.cond(p2_inside_anim_frame <= 0, lambda: 1, lambda: p2_inside_new_dir)
        p2_inside_new_frame = p2_inside_anim_frame + p2_inside_new_dir

        # Update if it has the ball, otherwise reset
        final_p2_inside_frame = jax.lax.select(p2_inside_has_ball, p2_inside_new_frame, 0)
        final_p2_inside_dir = jax.lax.select(p2_inside_has_ball, p2_inside_new_dir, 1)

        updated_p2_inside = updated_p2_inside.replace(
            animation_frame=final_p2_inside_frame,
            animation_direction=final_p2_inside_dir
        )

        # --- Animation for Player 2 Outside ---
        p2_outside_anim_frame = updated_p2_outside.animation_frame
        p2_outside_anim_dir = updated_p2_outside.animation_direction

        # Calculate next frame if it has the ball
        p2_outside_new_dir = jax.lax.cond(p2_outside_anim_frame >= 9, lambda: -1, lambda: p2_outside_anim_dir)
        p2_outside_new_dir = jax.lax.cond(p2_outside_anim_frame <= 0, lambda: 1, lambda: p2_outside_new_dir)
        p2_outside_new_frame = p2_outside_anim_frame + p2_outside_new_dir

        # Update if it has the ball, otherwise reset
        final_p2_outside_frame = jax.lax.select(p2_outside_has_ball, p2_outside_new_frame, 0)
        final_p2_outside_dir = jax.lax.select(p2_outside_has_ball, p2_outside_new_dir, 1)

        updated_p2_outside = updated_p2_outside.replace(
            animation_frame=final_p2_outside_frame,
            animation_direction=final_p2_outside_dir
        )

        # --- Ball Logic ---
        passer_id = state.controlled_player_id
        receiver_id = jax.lax.select(is_p1_inside_controlled, PlayerID.PLAYER1_OUTSIDE, PlayerID.PLAYER1_INSIDE)

        passer_x = jax.lax.select(is_p1_inside_controlled, updated_p1_inside.x, updated_p1_outside.x)
        passer_y = jax.lax.select(is_p1_inside_controlled, updated_p1_inside.y, updated_p1_outside.y)
        receiver_x = jax.lax.select(is_p1_inside_controlled, updated_p1_outside.x, updated_p1_inside.x)
        receiver_y = jax.lax.select(is_p1_inside_controlled, updated_p1_outside.y, updated_p1_inside.y)

        # 1. Passing
        can_pass = (state.ball.holder == passer_id)
        is_passing = jnp.any(jnp.asarray(action) == jnp.asarray(list(self._PASS_ACTIONS))) & can_pass

        passer_pos = jnp.array([passer_x, passer_y], dtype=jnp.float32)
        receiver_pos = jnp.array([receiver_x, receiver_y], dtype=jnp.float32)
        direction = receiver_pos - passer_pos
        norm = jnp.sqrt(jnp.sum(direction**2))
        safe_norm = jnp.where(norm == 0, 1.0, norm)
        ball_speed = 8.0
        vel = (direction / safe_norm) * ball_speed

        ball_state = jax.lax.cond(
            is_passing,
            lambda s: s.ball.replace(
                x=passer_x.astype(jnp.float32),
                y=passer_y.astype(jnp.float32),
                vel_x=vel[0],
                vel_y=vel[1],
                holder=PlayerID.NONE
            ),
            lambda s: s.ball,
            state
        )

        # 2. Ball movement if in flight
        ball_in_flight = (ball_state.holder == PlayerID.NONE)
        
        new_ball_x = ball_state.x + ball_state.vel_x
        new_ball_y = ball_state.y + ball_state.vel_y

        ball_state = jax.lax.cond(
            ball_in_flight,
            lambda b: b.replace(x=new_ball_x, y=new_ball_y),
            lambda b: b,
            ball_state
        )

        # --- Interception and Catching Logic ---
        # Order matters here. We check for interceptions before checking for a successful catch.

        # 3a. Interception by player2_inside
        ball_in_flight = (ball_state.holder == PlayerID.NONE)
        dist_to_p2_inside = jnp.sqrt((ball_state.x - updated_p2_inside.x)**2 + (ball_state.y - updated_p2_inside.y)**2)
        interception_radius = 5.0 # A bit smaller than player sprite, tune if needed
        is_intercepted_by_p2_inside = ball_in_flight & (dist_to_p2_inside < interception_radius)

        ball_state = jax.lax.cond(
            is_intercepted_by_p2_inside,
            lambda b: b.replace(holder=PlayerID.PLAYER2_INSIDE, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            ball_state
        )

        # 3b. Interception by player2_outside
        ball_in_flight = (ball_state.holder == PlayerID.NONE) # Re-evaluate in case of previous interception
        dist_to_p2_outside = jnp.sqrt((ball_state.x - updated_p2_outside.x)**2 + (ball_state.y - updated_p2_outside.y)**2)
        is_intercepted_by_p2_outside = ball_in_flight & (dist_to_p2_outside < interception_radius)

        ball_state = jax.lax.cond(
            is_intercepted_by_p2_outside,
            lambda b: b.replace(holder=PlayerID.PLAYER2_OUTSIDE, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            ball_state
        )

        # 3c. Ball catching by teammate
        ball_in_flight = (ball_state.holder == PlayerID.NONE) # Re-evaluate again
        dist_to_receiver = jnp.sqrt((ball_state.x - receiver_x)**2 + (ball_state.y - receiver_y)**2)
        catch_radius = 5.0
        is_caught = ball_in_flight & (dist_to_receiver < catch_radius)

        ball_state = jax.lax.cond(
            is_caught,
            lambda b: b.replace(holder=receiver_id, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            ball_state
        )

        # 4. If ball is held, its position should follow the holder
        def update_ball_pos_if_held(b_state, player_state, player_id):
            return jax.lax.cond(
                b_state.holder == player_id,
                lambda: b_state.replace(x=player_state.x.astype(jnp.float32), y=player_state.y.astype(jnp.float32)),
                lambda: b_state
            )
        
        ball_state = update_ball_pos_if_held(ball_state, updated_p1_inside, PlayerID.PLAYER1_INSIDE)
        ball_state = update_ball_pos_if_held(ball_state, updated_p1_outside, PlayerID.PLAYER1_OUTSIDE)
        ball_state = update_ball_pos_if_held(ball_state, updated_p2_inside, PlayerID.PLAYER2_INSIDE)
        ball_state = update_ball_pos_if_held(ball_state, updated_p2_outside, PlayerID.PLAYER2_OUTSIDE)

        # 5. Update controlled player
        is_p1_team_holder = (ball_state.holder == PlayerID.PLAYER1_INSIDE) | (ball_state.holder == PlayerID.PLAYER1_OUTSIDE)
        new_controlled_player_id = jax.lax.select(is_p1_team_holder, ball_state.holder, state.controlled_player_id)

        new_state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
            ball=ball_state,
            controlled_player_id=new_controlled_player_id,
        )

        observation = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return observation, new_state, reward, done, info

    def _get_reward(self, previous_state: DunkGameState, state: DunkGameState) -> float:
        """
        Calculates the reward from the environment state.
        """
        # Placeholder: return 0 reward for now
        return 0.0

    def _get_done(self, state: DunkGameState) -> bool:
        """
        Determines if the environment state is a terminal state
        """
        # Placeholder: game is never done for now
        return False

    def _get_info(self, state: DunkGameState) -> DunkInfo:
        """
        Extracts information from the environment state.
        """
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

        # For now, we'll just set up a basic background.
        # We'll need to create a background.npy file later.
        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'player', 'type': 'group', 'files': [f'player_{i}.npy' for i in range(10)]},
            {'name': 'player_no_ball', 'type': 'single', 'file': 'player_no_ball.npy'},
            {'name': 'enemy', 'type': 'group', 'files': [f'enemy_{i}.npy' for i in range(10)]},
            {'name': 'enemy_no_ball', 'type': 'single', 'file': 'enemy_no_ball.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
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

        # --- Sort players by visual Y position ---
        visual_ys = all_players_y - all_players_z
        sort_indices = jnp.argsort(visual_ys)

        # --- Render players in sorted order ---
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
        
        # --- Render Ball if in flight ---
        ball_in_flight = (state.ball.holder == PlayerID.NONE)
        ball_mask = self.SHAPE_MASKS['ball']

        def render_ball_body(current_raster):
            # x and y might be floats, so we cast to int for rendering
            ball_x = jnp.round(state.ball.x).astype(jnp.int32)
            ball_y = jnp.round(state.ball.y).astype(jnp.int32)
            return self.jr.render_at(current_raster, ball_x, ball_y, ball_mask)

        raster = jax.lax.cond(
            ball_in_flight,
            render_ball_body,
            lambda r: r,
            raster
        )

        return self.jr.render_from_palette(raster, self.PALETTE)