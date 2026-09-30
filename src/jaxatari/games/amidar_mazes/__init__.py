"""Amidar maze variants.

Default (package root) re-exports the original maze so existing imports keep working:
    import jaxatari.games.amidar_mazes as chosen_maze

Variants:
    from jaxatari.games.amidar_mazes import no_enemies
"""

from jaxatari.games.amidar_mazes.original import *  # noqa: F401,F403
