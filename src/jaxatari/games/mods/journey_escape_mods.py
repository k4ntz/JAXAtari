import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.journey_escape.journey_escape_mod_plugins import BackgroundStaticMod, SpeedUpPlayerMod, SpeedUpObstaclesMod, ReducePlayerSizeMod, RestrictPlayerMovementMod, ObstacleDiagonalMovementMod, ObstacleSteepDiagonalMovementMod, ObstacleAcceleratingBounceMod, ObstacleRandomDirectionSwitchMod, ObstacleChaoticMovementMod


class JourneyEscapeEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Journey Escape.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    # TODO: Confirm final modificaiton choices
    REGISTRY = {
        "background_static": BackgroundStaticMod,
        "speed_up_player": SpeedUpPlayerMod,
        "speed_up_obstacles": SpeedUpObstaclesMod,
        "reduce_player_size": ReducePlayerSizeMod,
        "restrict_player_movement": RestrictPlayerMovementMod,
        "obstacle_diagonal_movement": ObstacleDiagonalMovementMod,
        "obstacle_steep_diagonal_movement": ObstacleSteepDiagonalMovementMod,
        "obstacle_accelerating_bounce": ObstacleAcceleratingBounceMod,
        "obstacle_random_direction": ObstacleRandomDirectionSwitchMod,
        "obstacle_chaotic_movement": ObstacleChaoticMovementMod
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "journey_escape", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )