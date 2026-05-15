from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.enduro.enduro_mod_plugins import SpeedAndXPosHudMod, StartInCurveMod, \
    StartInMaxCurveMod, FilledRoadMod, SnowWeatherMod, NightWeatherMod, FogWeatherMod, DayWeatherMod, \
        SunsetWeatherMod, DawnWeatherMod, ShortDaysMod, NoOpponentsMod

class EnduroEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Enduro.
    """
    REGISTRY = {
        "hud": SpeedAndXPosHudMod,
        "start_in_curve": StartInCurveMod,
        "start_in_max_curve": StartInMaxCurveMod,
        "filled_road": FilledRoadMod,
        "snow": SnowWeatherMod,
        "night": NightWeatherMod,
        "fog": FogWeatherMod,
        "day": DayWeatherMod,
        "sunset": SunsetWeatherMod,
        "dawn": DawnWeatherMod,
        "short_days": ShortDaysMod,
        "no_opponents": NoOpponentsMod,
    }

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
