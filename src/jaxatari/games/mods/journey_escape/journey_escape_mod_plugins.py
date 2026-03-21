import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_journey_escape import JourneyEscapeState


class BackgroundStaticMod(JaxAtariInternalModPlugin):
    """Makes the background static (disables background animation)"""
    # TODO: implement


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
    # TODO: implement


class RestrictPlayerMovementMod(JaxAtariPostStepModPlugin):
    """Restricts players movement to four directions (up, down, left or right), disabling diagonal movement"""
   # TODO: implement


class ObstacleDiagonalMovementMod(JaxAtariInternalModPlugin):
    """
    Level 2: Obstacles move diagonally and bounce off walls.
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
    """Level 3: Obstacles move diagonally at steeper angle and bounce off walls.
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
    """Level 4: Obstacles alternate between slow and fast horizontal speed on each wall bounce."""
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
    """Level 5: Obstacles move diagonally at 1px/frame and randomly flip direction each frame."""
    constants_overrides = {
        "obstacle_horizontal_move_period": 1,       # always 1px/frame
        "diagonal_random_switch_prob": 0.3,          # 30% chance per frame per group

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