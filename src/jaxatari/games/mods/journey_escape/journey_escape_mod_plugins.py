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
    """Level 2: Obstacles move diagonally and bounce off walls."""
    # TODO: implement
   

class ObstacleSteepDiagonalMovementMod(JaxAtariInternalModPlugin):
    """Level 3/4: Obstacles move diagonally at steeper angle and bounce off walls"""
    # TODO: implement
    

class ObstacleRandomDirectionSwitchMod(JaxAtariInternalModPlugin):
    """
       Level 5: Obstacles move diagonally and bounce off walls,
        and randomly switch horizontal direction (left <-> right)
    """
    # TODO: implement