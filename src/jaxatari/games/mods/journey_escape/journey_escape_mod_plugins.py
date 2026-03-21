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
    # TODO: implement


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