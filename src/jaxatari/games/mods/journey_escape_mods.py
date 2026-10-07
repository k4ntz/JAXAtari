import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.journey_escape.journey_escape_mod_plugins import (
    BackgroundStaticMod,
    SpeedUpPlayerMod,
    SpeedUpObstaclesMod,
    ReducePlayerSizeMod,
    RestrictPlayerMovementMod,
    ObstacleDiagonalMovementMod,
    ObstacleSteepDiagonalMovementMod,
    ObstacleAcceleratingBounceMod,
    ObstacleRandomDirectionSwitchMod,
    ObstacleChaoticMovementMod,
)


class JourneyEscapeEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Journey Escape.

    Level-pattern mods skip the base env ahead to that difficulty while the
    concurrent env still progresses via the Scarab escape vehicle.
    """

    REGISTRY = {
        "background_static": BackgroundStaticMod,
        "speed_up_player": SpeedUpPlayerMod,
        "speed_up_obstacles": SpeedUpObstaclesMod,
        "reduce_player_size": ReducePlayerSizeMod,
        "restrict_player_movement": RestrictPlayerMovementMod,
        # Level shortcuts (match concurrent difficulty_level 1..4 / chaos)
        "obstacle_diagonal_movement": ObstacleDiagonalMovementMod,
        "obstacle_steep_diagonal_movement": ObstacleSteepDiagonalMovementMod,
        "obstacle_accelerating_bounce": ObstacleAcceleratingBounceMod,
        "obstacle_random_direction": ObstacleRandomDirectionSwitchMod,
        "obstacle_chaotic_movement": ObstacleChaoticMovementMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "journey_escape", "sprites")

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
