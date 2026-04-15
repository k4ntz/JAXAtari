import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin

class StartInCurveMod(JaxAtariInternalModPlugin):
    """
    Starts the game directly in the curve by setting straight_km_start to 0.
    """
    constants_overrides = {
        "straight_km_start": 0.0
    }

class SpeedAndXPosHudMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _render_cars_to_pass(self, raster: jnp.ndarray, state) -> jnp.ndarray:
        # Call original renderer method
        from jaxatari.games.jax_enduro2 import Enduro2Renderer
        raster = Enduro2Renderer._render_cars_to_pass(self._env.renderer, raster, state)
        
        # Now add speed and X position HUD at the top
        digit_sprites = self._env.renderer.SHAPE_MASKS['digits_black']
        spacing = digit_sprites.shape[2] + 2
        
        # 1. Speed (top left)
        speed = state.player_speed.astype(jnp.int32)
        s_hundreds = (speed // 100) % 10
        s_tens = (speed // 10) % 10
        s_ones = speed % 10
        
        raster = jax.lax.cond(
            speed >= 100,
            lambda r: self._env.renderer.jr.render_at(r, 10, 5, digit_sprites[s_hundreds]),
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            speed >= 10,
            lambda r: self._env.renderer.jr.render_at(r, 10 + spacing, 5, digit_sprites[s_tens]),
            lambda r: r,
            raster
        )
        
        raster = self._env.renderer.jr.render_at(raster, 10 + 2 * spacing, 5, digit_sprites[s_ones])
        
        # 2. X Position (top right)
        x_pos = state.player_x.astype(jnp.int32)
        x_hundreds = (x_pos // 100) % 10
        x_tens = (x_pos // 10) % 10
        x_ones = x_pos % 10
        
        start_x = 120
        
        raster = jax.lax.cond(
            x_pos >= 100,
            lambda r: self._env.renderer.jr.render_at(r, start_x, 5, digit_sprites[x_hundreds]),
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            x_pos >= 10,
            lambda r: self._env.renderer.jr.render_at(r, start_x + spacing, 5, digit_sprites[x_tens]),
            lambda r: r,
            raster
        )
        
        raster = self._env.renderer.jr.render_at(raster, start_x + 2 * spacing, 5, digit_sprites[x_ones])
        
        return raster
