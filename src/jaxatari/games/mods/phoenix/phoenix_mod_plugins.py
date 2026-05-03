from jaxatari.modification import JaxAtariInternalModPlugin


class BossLateMissilesMod(JaxAtariInternalModPlugin):
    """
    Make boss missiles appear a few pixels after spawn so they are visible
    later (e.g., after leaving dense boss-block area).
    """

    constants_overrides = {
        "BOSS_PROJECTILE_RENDER_DELAY_PX": 8,
    }

