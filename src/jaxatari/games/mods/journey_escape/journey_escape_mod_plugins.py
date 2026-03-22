import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_journey_escape import JourneyEscapeState


class BackgroundStaticMod(JaxAtariPostStepModPlugin):
    """Makes the background static (disables background animation)"""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: JourneyEscapeState, new_state: JourneyEscapeState) -> JourneyEscapeState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        return new_state.replace(bg_frames=prev_state.bg_frames)


class SpeedUpPlayerMod(JaxAtariInternalModPlugin):
    """Increases the player's movement speed"""
    
    constants_overrides = {
        "player_speed": 2
    }


class SpeedUpObstaclesMod(JaxAtariInternalModPlugin):
    """Increase the obstacles's movement speed"""

    constants_overrides = {
        "obstacle_speed_px_per_frame": 2
    }


class ReducePlayerSizeMod(JaxAtariInternalModPlugin):
    """Reduced the size of the player sprite."""
    
    # Player size reduced by 40%. New size: (5x15)
    asset_overrides = {
        "player": {
            'name': 'player',
            'type': 'group',
            'files': ['smaller_player_walk_front_1.npy', 'smaller_player_walk_front_0.npy',
                      'smaller_player_run_right_0.npy', 'smaller_player_run_right_1.npy',
                      'smaller_player_run_left_0.npy', 'smaller_player_run_left_1.npy']
        }
    }

    constants_overrides = {
        "player_width": 5,
        "player_height": 15
    }


class RestrictPlayerMovementMod(JaxAtariPostStepModPlugin):
    """Restricts players movement to four directions (up, down, left or right), disabling diagonal movement"""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: JourneyEscapeState, new_state: JourneyEscapeState) -> JourneyEscapeState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        diagonal_movement = (prev_state.player_y != new_state.player_y) & (prev_state.player_x != new_state.player_x)

        new_player_y = jnp.where(
            diagonal_movement,
            prev_state.player_y,    # if user tries to move diagonal, then only update movement in x-direction
            new_state.player_y
        ).astype(jnp.int32)
        return new_state.replace(player_y=new_player_y)


class ObstacleDiagonalMovementMod(JaxAtariInternalModPlugin):
    """
    Inspired by Level 2: Obstacles move diagonally and bounce off walls.
    """
    constants_overrides = {
        "obstacle_horizontal_move_period": 2,  # 1px/frame (try 2 for 0.5px/frame)

        # 0.0 = always vertical, 1.0 = always diagonal.
        "diagonal_probabilities": (
            0.0,     # 0:  barriers -- always vertical
            0.75,    # 1:  roadies (2, wide)
            0.75,    # 2:  big roadie (1)
            0.75,    # 3:  groupies (1)
            0.75,    # 4:  groupies (2, wide)
            0.75,    # 5:  groupies (3, tight)
            0.75,    # 6:  groupies (3, wide)
            0.75,    # 7:  big groupies (1)
            0.75,    # 8:  promoter (1)
            0.75,    # 9:  promoter (3, tight)
            0.75,    # 10: promoter (2, wide)
            0.75,    # 11: big promoter (1)
            0.75,    # 12: photographers (3)
            0.75,    # 13: photographers (2)
            0.75,    # 14: big photographer (1)
            0.75,    # 15: big manager (1)
        ),
    }

class ObstacleSteepDiagonalMovementMod(JaxAtariInternalModPlugin):
    """
    Inspired by Level 3: Obstacles move diagonally at steeper angle and bounce off walls.
    """
    constants_overrides = {
        "obstacle_horizontal_move_period": 1,  # 1px/frame (try 2 for 0.5px/frame)

        # 0.0 = always vertical, 1.0 = always diagonal.
        "diagonal_probabilities": (
            0.0,  # 0:  barriers -- always vertical
            0.75,  # 1:  roadies (2, wide)
            0.75,  # 2:  big roadie (1)
            0.75,  # 3:  groupies (1)
            0.75,  # 4:  groupies (2, wide)
            0.75,  # 5:  groupies (3, tight)
            0.75,  # 6:  groupies (3, wide)
            0.75,  # 7:  big groupies (1)
            0.75,  # 8:  promoter (1)
            0.75,  # 9:  promoter (3, tight)
            0.75,  # 10: promoter (2, wide)
            0.75,  # 11: big promoter (1)
            0.75,  # 12: photographers (3)
            0.75,  # 13: photographers (2)
            0.75,  # 14: big photographer (1)
            0.75,  # 15: big manager (1)
        ),
    }

class ObstacleAcceleratingBounceMod(JaxAtariInternalModPlugin):
    """
    Inspired by Level 4: Obstacles alternate between slow and fast horizontal speed on each wall bounce.
    """
    constants_overrides = {
        "obstacle_horizontal_move_period": 2,  # start slow (0.5 px/frame)
        "diagonal_speed_alternates": True,     # toggle to 1px/frame on each bounce

        # 0.0 = always vertical, 1.0 = always diagonal.
        "diagonal_probabilities": (
            0.0,   # 0:  barriers -- always vertical
            0.75,  # 1:  roadies (2, wide)
            0.75,  # 2:  big roadie (1)
            0.75,  # 3:  groupies (1)
            0.75,  # 4:  groupies (2, wide)
            0.75,  # 5:  groupies (3, tight)
            0.75,  # 6:  groupies (3, wide)
            0.75,  # 7:  big groupies (1)
            0.75,  # 8:  promoter (1)
            0.75,  # 9:  promoter (3, tight)
            0.75,  # 10: promoter (2, wide)
            0.75,  # 11: big promoter (1)
            0.75,  # 12: photographers (3)
            0.75,  # 13: photographers (2)
            0.75,  # 14: big photographer (1)
            0.75,  # 15: big manager (1)
        ),
    }

class ObstacleRandomDirectionSwitchMod(JaxAtariInternalModPlugin):
    """
    Inspired by Level 5: Obstacles move diagonally at 1px/frame and randomly flip direction each frame.
    """
    constants_overrides = {
        "obstacle_horizontal_move_period": 2,        # 1px/frame (try 2 for 0.5px/frame)
        "diagonal_random_switch_prob": 0.4,
        "diagonal_random_switch_cooldown": 20,       # base cooldown frames
        "diagonal_random_switch_cooldown_range": 40,  # actual cooldown = 20 + random(0..40)

        # 0.0 = always vertical, 1.0 = always diagonal.
        "diagonal_probabilities": (
            0.0,   # 0:  barriers -- always vertical
            0.75,  # 1:  roadies (2, wide)
            0.75,  # 2:  big roadie (1)
            0.75,  # 3:  groupies (1)
            0.75,  # 4:  groupies (2, wide)
            0.75,  # 5:  groupies (3, tight)
            0.75,  # 6:  groupies (3, wide)
            0.75,  # 7:  big groupies (1)
            0.75,  # 8:  promoter (1)
            0.75,  # 9:  promoter (3, tight)
            0.75,  # 10: promoter (2, wide)
            0.75,  # 11: big promoter (1)
            0.75,  # 12: photographers (3)
            0.75,  # 13: photographers (2)
            0.75,  # 14: big photographer (1)
            0.75,  # 15: big manager (1)
        ),
    }

class ObstacleChaoticMovementMod(JaxAtariInternalModPlugin):
    """
    Chaos Mode: Obstacles move diagonally and can individually switch direction,
    breaking away from their spawn group. Slightly higher spawn rate adds to the chaos.
    """
    constants_overrides = {
        "obstacle_horizontal_move_period": 2,        # 0.5px/frame
        "diagonal_random_switch_prob": 0.8,
        "diagonal_random_switch_cooldown": 10,       # base cooldown frames
        "diagonal_random_switch_cooldown_range": 40,  # actual cooldown = 20 + random(0..40)
        "diagonal_switch_per_obstacle": True,         # obstacles switch individually, not as group
        "row_spawn_period_frames": 19,                # faster spawning (default 50)

        # 0.0 = always vertical, 1.0 = always diagonal.
        "diagonal_probabilities": (
            0.0,   # 0:  barriers -- always vertical
            0.95,  # 1:  roadies (2, wide)
            0.95,  # 2:  big roadie (1)
            0.95,  # 3:  groupies (1)
            0.95,  # 4:  groupies (2, wide)
            0.95,  # 5:  groupies (3, tight)
            0.95,  # 6:  groupies (3, wide)
            0.95,  # 7:  big groupies (1)
            0.95,  # 8:  promoter (1)
            0.95,  # 9:  promoter (3, tight)
            0.95,  # 10: promoter (2, wide)
            0.95,  # 11: big promoter (1)
            0.95,  # 12: photographers (3)
            0.95,  # 13: photographers (2)
            0.95,  # 14: big photographer (1)
            0.95,  # 15: big manager (1)
        ),
    }