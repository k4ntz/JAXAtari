import os
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.seaquest.seaquest_mod_plugins import DisableEnemiesMod, NoDiversMod, EnemyMinesMod, FireBallsMod, UnlimitedOxygenMod, GravityMod, RandomColorEnemiesMod

class SeaquestEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Seaquest.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "disable_enemies": DisableEnemiesMod,
        "no_divers": NoDiversMod,
        "fireballs": FireBallsMod,
        # "peaceful_enemies": PeacefulEnemiesMod,
        # "lethal_divers": LethalDiversMod,
        "unlimited_oxygen": UnlimitedOxygenMod,
        "gravity": GravityMod,
        "random_color_enemies": RandomColorEnemiesMod,
        # "polluted_water": PollutedWaterMod,
        "mines": EnemyMinesMod,
        # "fireball": ReplaceTorpedoWithFireBallMod
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "seaquest", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):
        self._has_random_color = "random_color_enemies" in mods_config
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )

    @partial(jax.jit, static_argnames=['self'])
    def render(self, state):
        if self._has_random_color:
            renderer = self._env.renderer
            jr = renderer.jr
            raster = renderer.BACKGROUND
            step_counter = state.step_counter

            # Player
            player_anim_idx = (step_counter % 12) // 4
            raster = jr.render_at(
                raster, state.player_x, state.player_y,
                renderer.SHAPE_MASKS['player_sub'][player_anim_idx],
                flip_horizontal=state.player_direction == self._env.consts.FACE_LEFT,
                flip_offset=renderer.FLIP_OFFSETS['player_sub']
            )
            
            torp = state.player_missile_position
            raster = jax.lax.cond(
                torp[2] != 0,
                lambda r: jr.render_at_clipped(r, torp[0], torp[1], renderer.SHAPE_MASKS['player_torp'],
                                            flip_horizontal=torp[2] == self._env.consts.FACE_LEFT),
                lambda r: r,
                raster
            )

            raster = renderer._draw_divers(raster, state)
            
            # Enemy Torpedoes
            raster = jax.lax.fori_loop(
                0, state.enemy_missile_positions.shape[0],
                lambda i, r: renderer.render_object_sequentially(r, state.enemy_missile_positions[i], renderer.SHAPE_MASKS['enemy_torp'][None, ...], jnp.zeros(2, dtype=jnp.int32), 0),
                raster
            )

            # Sharks - Custom Color Logic
            shark_anim_idx = jax.lax.select((step_counter % 24) < 16, 0, 1)
            base_shark_masks = renderer.SHAPE_MASKS['shark_base']
            
            def draw_shark(i, r):
                hash_val = (i * 17 + (step_counter // 200)) % 8
                shark_color_id = renderer.SHARK_COLOR_MAP[hash_val]
                recolored_shark_masks = jnp.where(base_shark_masks != jr.TRANSPARENT_ID, shark_color_id, base_shark_masks)
                return renderer.render_object_sequentially(r, state.shark_positions[i], recolored_shark_masks, renderer.FLIP_OFFSETS['shark_base'], shark_anim_idx)
                
            raster = jax.lax.fori_loop(0, state.shark_positions.shape[0], draw_shark, raster)
            
            # UI Elements
            score_digits = jr.int_to_digits(state.score, max_digits=6)
            raster = jr.render_label(raster, 58, 18, score_digits, renderer.SHAPE_MASKS['digits'], spacing=8, max_digits=6)
            
            raster = jr.render_indicator(raster, 14, 28, state.lives, renderer.SHAPE_MASKS['life_indicator'], spacing=10, max_value=3)
            raster = jr.render_indicator(raster, 49, 178, state.divers_collected, renderer.SHAPE_MASKS['diver_indicator'], spacing=10, max_value=6)

            raster = jr.render_bar(raster, 49, 170, state.oxygen, 64, 63, 5, renderer.OXYGEN_COLOR_ID, jr.TRANSPARENT_ID)

            raster = jr.draw_rects(
                raster,
                positions=jnp.array([[0, 0]]),
                sizes=jnp.array([[8, renderer.config.game_dimensions[0]]]),
                color_id=renderer.BACKGROUND[0, 0]
            )
            
            return jr.render_from_palette(raster, renderer.PALETTE)

        return self._env.render(state)
