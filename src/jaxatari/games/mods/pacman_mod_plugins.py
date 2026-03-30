import os
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jaxatari.games.jax_pacman import PacmanState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


# Level order when using multi_maze_campaign (geometry: {name}.txt, pellets: {name}_pellet.txt if present).
DEFAULT_PACMAN_MAZE_LEVEL_BASENAMES: Tuple[str, ...] = (
    "maze_atari",
    "maze1",
    "maze2",
    "maze3",
    "maze4",
)


def resolve_pacman_maze_level_specs(
    pacman_maps_dir: str,
    basenames: Tuple[str, ...] = DEFAULT_PACMAN_MAZE_LEVEL_BASENAMES,
) -> List[Tuple[str, str]]:
    """Build (geometry_path, pellet_layout_path) for each existing maze file."""
    specs: List[Tuple[str, str]] = []
    for base in basenames:
        geom = os.path.join(pacman_maps_dir, f"{base}.txt")
        if not os.path.isfile(geom):
            continue
        pellet = os.path.join(pacman_maps_dir, f"{base}_pellet.txt")
        if not os.path.isfile(pellet):
            pellet = geom
        specs.append((geom, pellet))
    return specs

# Part 1: Simple Modifications

class FasterPacmanMod(JaxAtariInternalModPlugin):
    """
    Faster Pac-Man (+20% Score):
    Change: Set PLAYER_SPEED=2 and add a 20% score multiplier (12 for dot, 60 for power).
    Effect: Makes Pac-Man highly agile, rewarding riskier, high-speed maneuvers.
    """
    # Simply override the constants at initialization
    constants_overrides = {
        "PLAYER_SPEED": 2,
        "PELLET_DOT_SCORE": 12,
        "PELLET_POWER_SCORE": 60
    }

class SlowerGhostsMod(JaxAtariInternalModPlugin):
    """
    Slower Ghosts (Easy Mode):
    Change: Set ghost speed to 0.5 (moves every other frame).
    Effect: Drastically reduces the threat level.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _ghost_step(self, state: PacmanState, keys: jax.Array) -> PacmanState:
        """
        Intercepts ghost updates. We only call the base logic if
        step_counter % 2 == 0; otherwise ghosts keep their previous state.
        """
        should_move = state.step_counter % 2 == 0
        return jax.lax.cond(
            should_move,
            lambda s: type(self._env)._ghost_step(self._env, s, keys),
            lambda s: s,
            state,
        )

class NoFrightMod(JaxAtariInternalModPlugin):
    """
    No Frightened State (Defensive Play):
    Change: Remove the power pellet effect completely.
    """
    # Overriding the duration to 0 means eating a power pellet grants score but instantly expires.
    constants_overrides = {
        "FRIGHTENED_DURATION": 0
    }

class HalfDotsMod(JaxAtariPostStepModPlugin):
    """
    Half Dots (Shorter Episodes):
    Change: Reduce total dots on map to ~120 by pre-collecting half of them.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        """
        Instantly collects half of the dots identically upon reset.
        """
        rng_key, pickup_key = jax.random.split(state.key)
        
        layout = self._env.consts.MAZE_LAYOUT
        valid_dots = (layout == 2)
        
        # Randomly select ~50% of the dots to be removed
        coin_flips = jax.random.bernoulli(pickup_key, p=0.5, shape=valid_dots.shape)
        dots_to_remove = jnp.logical_and(valid_dots, coin_flips)
        
        newly_removed = jnp.logical_and(state.pellets_collected == 0, dots_to_remove)
        new_pellets_collected = jnp.where(
            newly_removed,
            jnp.array(1, dtype=jnp.int32),
            state.pellets_collected,
        ).astype(jnp.int32)

        removed_count = jnp.sum(newly_removed)
        new_dots_remaining = state.dots_remaining - removed_count.astype(jnp.int32)
        
        new_state = state._replace(
            key=rng_key,
            pellets_collected=new_pellets_collected,
            dots_remaining=new_dots_remaining
        )
        
        return obs, new_state


class RandomStartMod(JaxAtariPostStepModPlugin):
    """
    Random Start Position (Robustness):
    Change: Pac-Man spawns at a random valid tile instead of the center.
    """
    def _find_nearest_node_idx_jax(self, x: jax.Array, y: jax.Array) -> jax.Array:
        node_x = jnp.asarray(self._env.node_positions_x, dtype=jnp.int32)
        node_y = jnp.asarray(self._env.node_positions_y, dtype=jnp.int32)
        dx = node_x - x.astype(jnp.int32)
        dy = node_y - y.astype(jnp.int32)
        dist_sq = dx * dx + dy * dy
        return jnp.argmin(dist_sq).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        rng_key, spawn_key = jax.random.split(state.key)
        
        layout = self._env.consts.MAZE_LAYOUT
        _, w = layout.shape
        valid_mask = (layout.flatten() == 0)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        flat_idx = jax.random.categorical(spawn_key, logits)
        
        spawn_y_tile = flat_idx // w
        spawn_x_tile = flat_idx % w
        spawn_px = (spawn_x_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        spawn_py = (spawn_y_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)

        spawn_node_idx = self._find_nearest_node_idx_jax(spawn_px, spawn_py)
        
        new_state = state._replace(
            key=rng_key,
            player_x=spawn_px.astype(jnp.int32),
            player_y=spawn_py.astype(jnp.int32),
            player_target_node_index=spawn_node_idx,
            player_current_node_index=spawn_node_idx
        )
        
        return obs, new_state


class LimitedVisionMod(JaxAtariInternalModPlugin):
    """
    Limited Vision / Fog of War:
    - Render applies a fog mask outside a radius around Pac-Man.
    - Ghost entries outside that radius are hidden in object observations.
    """
    VISION_RADIUS_PX = 36  # half of previous 72
    FOG_COLOR = (40, 40, 40)  # use (35, 35, 35) for dark gray instead of pure black
    FOG_ALPHA = 255  # 255 = fully opaque fog outside radius

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array):
        renderer = self._env.renderer

        if not getattr(renderer, "_limited_vision_fog_installed", False):
            base_render = renderer.render
            radius_px = self.VISION_RADIUS_PX
            fog_color_rgb = self.FOG_COLOR
            fog_alpha_value = self.FOG_ALPHA

            def fog_render(state: PacmanState):
                frame = base_render(state)

                h, w, _ = frame.shape
                yy = jnp.arange(h, dtype=jnp.int32)[:, None]
                xx = jnp.arange(w, dtype=jnp.int32)[None, :]

                # Only apply fog to the maze band; keep HUD rows readable.
                maze_y_min = self._env.consts.TILE_SIZE * 4
                maze_y_max = maze_y_min + (self._env.consts.MAZE_HEIGHT * self._env.consts.TILE_SIZE)
                in_maze_band = jnp.logical_and(yy >= maze_y_min, yy < maze_y_max)
                in_maze_band = in_maze_band[..., None]

                center_x = state.player_x - 4
                center_y = state.player_y

                dx = xx - center_x
                dy = yy - center_y
                radius_sq = jnp.array(radius_px * radius_px, dtype=jnp.int32)
                visible = (dx * dx + dy * dy) <= radius_sq
                visible = visible[..., None]

                base = frame.astype(jnp.uint16)
                fog_color = jnp.array(fog_color_rgb, dtype=jnp.uint16)
                fog_alpha = jnp.array(fog_alpha_value, dtype=jnp.uint16)
                fogged = ((base * (255 - fog_alpha) + fog_color * fog_alpha) // 255).astype(jnp.uint8)
                apply_fog = jnp.logical_and(in_maze_band, jnp.logical_not(visible))
                return jnp.where(apply_fog, fogged, frame)

            renderer.render = fog_render
            renderer._limited_vision_fog_installed = True

        return type(self._env).reset(self._env, key)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PacmanState):
        obs = type(self._env)._get_observation(self._env, state)

        ghosts = obs.ghosts
        dx = ghosts[:, 0] - state.player_x
        dy = ghosts[:, 1] - state.player_y
        dist_sq = dx * dx + dy * dy
        max_dist_sq = jnp.array(self.VISION_RADIUS_PX * self.VISION_RADIUS_PX, dtype=jnp.int32)
        visible = dist_sq <= max_dist_sq

        hidden_row = jnp.zeros((1, ghosts.shape[1]), dtype=ghosts.dtype)
        hidden_ghosts = jnp.repeat(hidden_row, ghosts.shape[0], axis=0)
        masked_ghosts = jnp.where(visible[:, None], ghosts, hidden_ghosts)

        return obs._replace(ghosts=masked_ghosts)

# Part 2: Difficult Modifications (Requires Base Upgrades)
class CoopMultiplayerMod(JaxAtariPostStepModPlugin):
    """
    Cooperative-style mode.
    Full multi-agent Pacman support requires broader base-env changes; this mod keeps
    gameplay JAX-safe by adding a support bonus (extra life) and randomizing spawn.
    """
    def _find_nearest_node_idx_jax(self, x: jax.Array, y: jax.Array) -> jax.Array:
        node_x = jnp.asarray(self._env.node_positions_x, dtype=jnp.int32)
        node_y = jnp.asarray(self._env.node_positions_y, dtype=jnp.int32)
        dx = node_x - x.astype(jnp.int32)
        dy = node_y - y.astype(jnp.int32)
        dist_sq = dx * dx + dy * dy
        return jnp.argmin(dist_sq).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        rng_key, spawn_key = jax.random.split(state.key)
        
        layout = self._env.consts.MAZE_LAYOUT
        _, w = layout.shape
        valid_mask = (layout.flatten() == 0)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        flat_idx = jax.random.categorical(spawn_key, logits)
        
        spawn_y_tile = flat_idx // w
        spawn_x_tile = flat_idx % w
        
        spawn_px = (spawn_x_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        spawn_py = (spawn_y_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        
        spawn_node_idx = self._find_nearest_node_idx_jax(spawn_px, spawn_py)
        
        new_state = state._replace(
            key=rng_key,
            lives=jnp.maximum(state.lives, jnp.array(2, dtype=jnp.int32)),
            player_x=spawn_px.astype(jnp.int32),
            player_y=spawn_py.astype(jnp.int32),
            player_current_node_index=spawn_node_idx.astype(jnp.int32),
            player_target_node_index=spawn_node_idx.astype(jnp.int32),
        )
        
        return self._env._get_observation(new_state), new_state

class MultiMazeCampaignMod(JaxAtariInternalModPlugin):
    """
    Multi-map campaign: clearing all pellets advances to the next preloaded maze (wrapped).
    After the last maze, the run wraps to the first map, level resets to 1, and score resets to 0.

    Requires at least two maze files for DEFAULT_PACMAN_MAZE_LEVEL_BASENAMES under pacmanMaps/;
    otherwise attach_to_env is a no-op (base env stays single-maze).
    """

    @staticmethod
    def attach_to_env(env) -> None:
        from jaxatari.games import jax_pacman as jpm

        if not isinstance(env, jpm.JaxPacman):
            return
        pacman_maps_dir = os.path.join(os.path.dirname(jpm.__file__), "pacmanMaps")
        level_specs = resolve_pacman_maze_level_specs(pacman_maps_dir)
        if len(level_specs) <= 1:
            return

        layouts = []
        vitamin_tiles: List[Tuple[int, int]] = []
        exp_h, exp_w = env.consts.MAZE_HEIGHT, env.consts.MAZE_WIDTH
        for _geom_path, pellet_path in level_specs:
            layout, vr, vc = env._parse_maze_layout_from_file(pellet_path)
            arr = np.asarray(layout)
            if arr.shape != (exp_h, exp_w):
                raise ValueError(
                    f"Maze layout shape {arr.shape} from {pellet_path} "
                    f"does not match ({exp_h}, {exp_w})"
                )
            layouts.append(layout)
            vitamin_tiles.append((vr, vc))

        env.reload_maze_campaign(level_specs, layouts, vitamin_tiles)