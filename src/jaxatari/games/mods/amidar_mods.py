"""
Wrappers for Amidar modifications.

MAZE-WRAPPERS have to be applied before any other wrappers:
Amidar bakes maze-dependent constants into JIT-compiled methods.
To safely switch mazes, these wrappers rebuild a fresh JaxAmidar instance with
the selected maze module, ensuring the JIT graph is retraced with the new shapes.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from jaxatari.wrappers import JaxatariWrapper


class _AmidarMazeWrapperBase(JaxatariWrapper):
    """Base wrapper that re-instantiates the Amidar env with a chosen maze module.

    Usage: Subclasses pass a specific maze module from `jaxatari.games.amidar_mazes`.
    """

    def __init__(self, env, maze_module, *, reward_funcs: Optional[list] = None):
        # Lazy imports to avoid circulars
        from jaxatari.games import jax_amidar as amidar_module

        # Prefer explicit reward_funcs, else preserve from the given env if present
        if reward_funcs is None:
            reward_funcs = getattr(env, "reward_funcs", None)

        # 1) Start from base constants
        c = amidar_module.AmidarConstants()

        # 2) Override maze-dependent fields from the provided module
        # Basics
        c = c._replace(
            PATH_THICKNESS_HORIZONTAL=maze_module.PATH_THICKNESS_HORIZONTAL,
            PATH_THICKNESS_VERTICAL=maze_module.PATH_THICKNESS_VERTICAL,
            INITIAL_PLAYER_POSITION=maze_module.INITIAL_PLAYER_POSITION,
            PLAYER_STARTING_PATH=maze_module.PLAYER_STARTING_PATH,
            MAX_ENEMIES=maze_module.MAX_ENEMIES,
            INITIAL_ENEMY_POSITIONS=maze_module.INITIAL_ENEMY_POSITIONS,
            PATH_CORNERS=maze_module.PATH_CORNERS,
            HORIZONTAL_PATH_EDGES=maze_module.HORIZONTAL_PATH_EDGES,
            VERTICAL_PATH_EDGES=maze_module.VERTICAL_PATH_EDGES,
            PATH_EDGES=maze_module.PATH_EDGES,
            RECTANGLES=maze_module.RECTANGLES,
            RECTANGLE_BOUNDS=maze_module.RECTANGLE_BOUNDS,
            CORNER_RECTANGLES=maze_module.CORNER_RECTANGLES,
            SHORT_PATHS=maze_module.SHORT_PATHS,
        )

        # 3) Recompute dependent fields
        # INITIAL_ENEMY_DIRECTIONS depends on MAX_ENEMIES
        right_dir = c.RIGHT
        c = c._replace(
            INITIAL_ENEMY_DIRECTIONS=jnp.array([right_dir] * int(c.MAX_ENEMIES)),
        )

        # PATH masks/patterns/sprites depend on path geometry and thickness
        PATH_MASK, RENDERING_PATH_MASK = amidar_module.generate_path_mask(
            c.WIDTH,
            c.HEIGHT,
            c.PATH_THICKNESS_HORIZONTAL,
            c.PATH_THICKNESS_VERTICAL,
            c.HORIZONTAL_PATH_EDGES,
            c.VERTICAL_PATH_EDGES,
            c.PATH_CORNERS,
            jnp.full((c.HORIZONTAL_PATH_EDGES.shape[0],), True),
            jnp.full((c.VERTICAL_PATH_EDGES.shape[0],), True),
            jnp.full((c.PATH_CORNERS.shape[0],), True),
        )

        PATH_PATTERN_BROWN, PATH_PATTERN_GREEN, WALKED_ON_PATTERN = amidar_module.generate_path_pattern(
            c.WIDTH, c.HEIGHT, c.PATH_COLOR_BROWN, c.PATH_COLOR_GREEN, c.WALKED_ON_COLOR
        )

        PATH_SPRITE_BROWN = jnp.where(
            RENDERING_PATH_MASK[:, :, None] == 1,
            PATH_PATTERN_BROWN,
            jnp.full((c.HEIGHT, c.WIDTH, 4), 0, dtype=jnp.uint8),
        )
        PATH_SPRITE_GREEN = jnp.where(
            RENDERING_PATH_MASK[:, :, None] == 1,
            PATH_PATTERN_GREEN,
            jnp.full((c.HEIGHT, c.WIDTH, 4), 0, dtype=jnp.uint8),
        )

        # AmidarConstants defines PATH_MASK, RENDERING_PATH_MASK, PATH_PATTERN_* and WALKED_ON_PATTERN
        # as class attributes (not NamedTuple fields). We cannot set them via _replace.
        # Assign on the class so instance lookup (which falls back to class attributes) sees the update.
        amidar_module.AmidarConstants.PATH_MASK = PATH_MASK
        amidar_module.AmidarConstants.RENDERING_PATH_MASK = RENDERING_PATH_MASK
        amidar_module.AmidarConstants.PATH_PATTERN_BROWN = PATH_PATTERN_BROWN
        amidar_module.AmidarConstants.PATH_PATTERN_GREEN = PATH_PATTERN_GREEN
        amidar_module.AmidarConstants.WALKED_ON_PATTERN = WALKED_ON_PATTERN

        # PATH_SPRITE_* are annotated fields in AmidarConstants, so keep them on the instance
        c = c._replace(
            PATH_SPRITE_BROWN=PATH_SPRITE_BROWN,
            PATH_SPRITE_GREEN=PATH_SPRITE_GREEN,
        )

        # 4) Build a fresh env so JIT sees a new constants object
        new_env = amidar_module.JaxAmidar(constants=c, reward_funcs=reward_funcs)

        super().__init__(new_env)
        self._env = new_env


class OriginalMaze(_AmidarMazeWrapperBase):
    """Use the original Amidar maze."""

    def __init__(self, env, *, reward_funcs: Optional[list] = None):
        from jaxatari.games.amidar_mazes import original as maze
        super().__init__(env, maze, reward_funcs=reward_funcs)


class NoEnemiesMaze(_AmidarMazeWrapperBase):
    """Use the original maze geometry but with enemies disabled in the layout module."""

    def __init__(self, env, *, reward_funcs: Optional[list] = None):
        from jaxatari.games.amidar_mazes import no_enemies as maze
        super().__init__(env, maze, reward_funcs=reward_funcs)


class SlightChangesNoEnemiesMaze(_AmidarMazeWrapperBase):
    """Use a slightly modified maze geometry without enemies."""

    def __init__(self, env, *, reward_funcs: Optional[list] = None):
        from jaxatari.games.amidar_mazes import slight_changes_no_enemies as maze
        super().__init__(env, maze, reward_funcs=reward_funcs)


class UpperRightCornerNoEnemiesMaze(_AmidarMazeWrapperBase):
    """Use a variant with the upper-right corner as the start point and no enemies."""

    def __init__(self, env, *, reward_funcs: Optional[list] = None):
        from jaxatari.games.amidar_mazes import upper_right_corner_no_enemies as maze
        super().__init__(env, maze, reward_funcs=reward_funcs)


class CustomAmidarMaze(_AmidarMazeWrapperBase):
    """Use a custom maze module.

    Pass any module that provides the expected attributes used by AmidarConstants,
    e.g. PATH_CORNERS, HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES, PATH_EDGES,
    RECTANGLES, RECTANGLE_BOUNDS, CORNER_RECTANGLES, SHORT_PATHS, INITIAL_* and sizes.
    """

    def __init__(self, env, maze_module, *, reward_funcs: Optional[list] = None):
        super().__init__(env, maze_module, reward_funcs=reward_funcs)


__all__ = [
    "OriginalMaze",
    "NoEnemiesMaze",
    "SlightChangesNoEnemiesMaze",
    "UpperRightCornerNoEnemiesMaze",
    "CustomAmidarMaze",
]
