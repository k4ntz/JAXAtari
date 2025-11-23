from jax import numpy as jnp
from typing import Tuple
import jax.lax
import chex
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from functools import partial

# =============================================================================
# TASK 1: Define Constants and Game State
# =============================================================================

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
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER1_COLOR: Tuple[int, int, int] = (200, 72, 72)
    PLAYER2_COLOR: Tuple[int, int, int] = (72, 72, 200)
    BALL_COLOR: Tuple[int, int, int] = (204, 102, 0)
    BALL_SIZE: Tuple[int, int] = (3,3)
    BALL_START: Tuple [int, int] = (100, 70)
    WALL_COLOR: Tuple[int, int, int] = (142, 142, 142)
    FIELD_COLOR: Tuple[int, int, int] = (128, 128, 128)
    JUMP_STRENGTH: int = 5 #adjustable if necessary and more of a placeholder value 
    PLAYER_MAX_SPEED: int = 4 #adjustable if necessary and more of a placeholder value
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
    controlled_player: chex.Array

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


    # =========================================================================
    # TASK 2: Implement Init and Reset
    # =========================================================================
    
    def _init_state(self) -> DunkGameState:
        """
        Creates the very first state of the game.
        Use values from self.constants.
        """
        return DunkGameState(
            player1_inside=PlayerState(x=125, y=100, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player1_outside=PlayerState(x=80, y=120, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player2_inside=PlayerState(x=170, y=100, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            player2_outside=PlayerState(x=200, y=120, vel_x=0, vel_y=0, z=0, vel_z=0, role=0, animation_frame=0, animation_direction=1),
            ball=BallState(x=0, y=0, vel_x=0, vel_y=0, holder=PlayerID.PLAYER1_INSIDE),
            player_score=0,
            enemy_score=0,
            step_counter=0,
            acceleration_counter=0,
            game_mode=GameMode.PLAY_SELECTION,
            offensive_play=OffensivePlay.NONE,
            defensive_play=DefensivePlay.NONE,
            play_step=0,
            controlled_player=PlayerID.PLAYER1_INSIDE,
        )



    def _get_player_action_effects(self, action: int, player_z: chex.Array, constants: DunkConstants) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Determines the velocity for 8-way movement and the impulse for Z-axis jumps.
        """
        # --- X/Y Movement on the ground plane ---
        is_moving_left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT) | \
                         (action == Action.LEFTFIRE) | (action == Action.UPLEFTFIRE) | (action == Action.DOWNLEFTFIRE)
        is_moving_right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT) | \
                          (action == Action.RIGHTFIRE) | (action == Action.UPRIGHTFIRE) | (action == Action.DOWNRIGHTFIRE)

        vel_x = jnp.array(0, dtype=jnp.int32)
        vel_x = jax.lax.select(is_moving_left, -constants.PLAYER_MAX_SPEED, vel_x)
        vel_x = jax.lax.select(is_moving_right, constants.PLAYER_MAX_SPEED, vel_x)

        is_moving_up = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT) | \
                       (action == Action.UPFIRE) | (action == Action.UPLEFTFIRE) | (action == Action.UPRIGHTFIRE)
        is_moving_down = (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT) | \
                         (action == Action.DOWNFIRE) | (action == Action.DOWNLEFTFIRE) | (action == Action.DOWNRIGHTFIRE)

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

    def _handle_passing(self, state: DunkGameState, action: int) -> DunkGameState:
        """Handles the logic for passing the ball between players."""
        is_pass_action = (action == Action.DOWNFIRE)

        # Determine the new controlled player and ball holder if a pass occurs
        new_controlled_player = jax.lax.select(
            is_pass_action & (state.controlled_player == PlayerID.PLAYER1_INSIDE),
            PlayerID.PLAYER1_OUTSIDE,
            state.controlled_player
        )
        new_controlled_player = jax.lax.select(
            is_pass_action & (state.controlled_player == PlayerID.PLAYER1_OUTSIDE),
            PlayerID.PLAYER1_INSIDE,
            new_controlled_player
        )

        # The ball holder becomes the newly controlled player
        new_ball_holder = jax.lax.select(is_pass_action, new_controlled_player, state.ball.holder)

        return state.replace(
            controlled_player=new_controlled_player,
            ball=state.ball.replace(holder=new_ball_holder)
        )

    # =========================================================================
    # TASK 3: Implement the Step Function
    # =========================================================================
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DunkGameState, action: int) -> Tuple[DunkObservation, DunkGameState, float, bool, DunkInfo]:
        """
        Takes an action in the game and returns the new game state.
        """
        # 1. Handle passing logic to update controlled player and ball holder
        state = self._handle_passing(state, action)

        # 2. Determine which player to update based on the (potentially new) controlled_player
        is_p1_inside_controlled = (state.controlled_player == PlayerID.PLAYER1_INSIDE)
        is_p1_outside_controlled = (state.controlled_player == PlayerID.PLAYER1_OUTSIDE)

        # Use a NOOP action for the player who is not controlled
        p1_inside_action = jax.lax.select(is_p1_inside_controlled, action, Action.NOOP)
        p1_outside_action = jax.lax.select(is_p1_outside_controlled, action, Action.NOOP)

        # 3. Update player states based on actions
        updated_p1_inside = self._update_player(state.player1_inside, p1_inside_action, self.constants)
        updated_p1_outside = self._update_player(state.player1_outside, p1_outside_action, self.constants)
        
        # Other players remain static for now
        updated_p2_inside = self._update_player(state.player2_inside, Action.NOOP, self.constants)
        updated_p2_outside = self._update_player(state.player2_outside, Action.NOOP, self.constants)

        # 4. Update animation for the player who has the ball
        p1_inside_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE)
        p1_outside_has_ball = (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)

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
        
        new_state = state.replace(
            player1_inside=updated_p1_inside,
            player1_outside=updated_p1_outside,
            player2_inside=updated_p2_inside,
            player2_outside=updated_p2_outside,
            # The ball holder and controlled player were already updated in _handle_passing
            # so we carry them over from the intermediate state.
            ball=state.ball,
            controlled_player=state.controlled_player,
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

import os

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

        # Determine which player has the ball
        p1_inside_has_ball = (state.ball.holder == PlayerID.PLAYER1_INSIDE)
        p1_outside_has_ball = (state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        
        # --- Draw Player 1 Inside ---
        p1_inside_state = state.player1_inside
        p1_inside_visual_y = p1_inside_state.y - p1_inside_state.z
        p1_inside_mask = jax.lax.select(
            p1_inside_has_ball,
            self.SHAPE_MASKS['player'][p1_inside_state.animation_frame],
            self.SHAPE_MASKS['player_no_ball']
        )
        raster = self.jr.render_at(raster, p1_inside_state.x, p1_inside_visual_y, p1_inside_mask)

        # --- Draw Player 1 Outside ---
        p1_outside_state = state.player1_outside
        p1_outside_visual_y = p1_outside_state.y - p1_outside_state.z
        p1_outside_mask = jax.lax.select(
            p1_outside_has_ball,
            self.SHAPE_MASKS['player'][p1_outside_state.animation_frame],
            self.SHAPE_MASKS['player_no_ball']
        )
        raster = self.jr.render_at(raster, p1_outside_state.x, p1_outside_visual_y, p1_outside_mask)

        # Draw other players (without ball)
        no_ball_mask = self.SHAPE_MASKS['player_no_ball']
        
        p2_inside_state = state.player2_inside
        p2_inside_visual_y = p2_inside_state.y - p2_inside_state.z
        raster = self.jr.render_at(raster, p2_inside_state.x, p2_inside_visual_y, no_ball_mask)

        p2_outside_state = state.player2_outside
        p2_outside_visual_y = p2_outside_state.y - p2_outside_state.z
        raster = self.jr.render_at(raster, p2_outside_state.x, p2_outside_visual_y, no_ball_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)