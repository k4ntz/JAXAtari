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
        # One probability per group:
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
    """Level 3: Obstacles move diagonally at steeper angle and bounce off walls"""
    # TODO: implement
    

class ObstacleRandomDirectionSwitchMod(JaxAtariInternalModPlugin):
    """
       Level 5: Obstacles move diagonally and bounce off walls,
        and randomly switch horizontal direction (left <-> right)
    """
    # TODO: implement