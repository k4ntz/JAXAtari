import os
import jax
import jax.numpy as jnp
from functools import partial
from jaxtari.modification import JaxtariModController
from jaxtari.games.mods.seaquest.seaquest_mod_plugins import (
    DisableEnemiesMod,
    NoDiversMod,
    EnemyMinesMod,
    FireBallsMod,
    UnlimitedOxygenMod,
    GravityMod,
    RandomColorEnemiesMod,
    PenalizeDiverShootingMod,
    ChangeBackgroundColorMod,
    PeacefulEnemiesMod,
    NoEnemyTorpedoesMod,
    FasterEnemiesMod,
    SlowerEnemiesMod,
    ShiftLanesMod,
    OnlySubmarinesMod,
    OnlySharksMod,
    VerticalOscillationMod,
    FastOxygenDrainMod,
    SlowOxygenDrainMod,
    SurfaceSubAlwaysMod,
    InvertedLanesMod,
    LethalDiversMod,
    ShootableDiversMod,
    PeacefulSharksOnlyMod,
    CollectSharksOnContactMod,
    DisableDiverCollectMod,
    NoSurfacingDeathMod,
    UnshootableSharksMod,
    InvertedControlsMod,
    MomentumMod,
    TinyDiversMod,
    DynamicLaneDriftMod,
    ContinuousRandomSpawnsMod,
)


class SeaquestEnvMod(JaxtariModController):
    """
    Game-specific Mod Controller for Seaquest.
    It simply inherits all logic from JaxtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        # Existing
        "disable_enemies": DisableEnemiesMod,
        "no_divers": NoDiversMod,
        "fireballs": FireBallsMod,
        "unlimited_oxygen": UnlimitedOxygenMod,
        "gravity": GravityMod,
        "random_color_enemies": RandomColorEnemiesMod,
        "penalize_diver_shooting": PenalizeDiverShootingMod,
        "mines": EnemyMinesMod,
        # Upstreamed robustness set
        "change_background_color": ChangeBackgroundColorMod,
        "peaceful_enemies": PeacefulEnemiesMod,
        "no_enemy_torpedoes": NoEnemyTorpedoesMod,
        "faster_enemies": FasterEnemiesMod,
        "slower_enemies": SlowerEnemiesMod,
        "shift_lanes": ShiftLanesMod,
        "only_submarines": OnlySubmarinesMod,
        "only_sharks": OnlySharksMod,
        "vertical_oscillation": VerticalOscillationMod,
        "fast_oxygen_drain": FastOxygenDrainMod,
        "slow_oxygen_drain": SlowOxygenDrainMod,
        "surface_sub_always": SurfaceSubAlwaysMod,
        "inverted_lanes": InvertedLanesMod,
        "lethal_divers": LethalDiversMod,
        "shootable_divers": ShootableDiversMod,
        "_peaceful_sharks": PeacefulSharksOnlyMod,
        "_collect_sharks": CollectSharksOnContactMod,
        "_disable_diver_collect": DisableDiverCollectMod,
        "_shootable_divers": ShootableDiversMod,
        "_unshootable_sharks": UnshootableSharksMod,
        "collectable_sharks": ["_peaceful_sharks", "_collect_sharks"],
        "swap_diver_shark_roles": [
            "_peaceful_sharks",
            "_collect_sharks",
            "_disable_diver_collect",
            "_unshootable_sharks",
            "_shootable_divers",
            "lethal_divers",
        ],
        "no_surfacing_death": NoSurfacingDeathMod,
        "inverted_controls": InvertedControlsMod,
        "momentum": MomentumMod,
        "tiny_divers": TinyDiversMod,
        "dynamic_lane_drift": DynamicLaneDriftMod,
        "continuous_random_spawns": ContinuousRandomSpawnsMod,
        "random_spawns": ContinuousRandomSpawnsMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "seaquest", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):
        self._has_random_color = "random_color_enemies" in mods_config
        self._has_change_background = "change_background_color" in mods_config
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
        if self._has_change_background:
            self._apply_change_background_color()

    def _reapply_renderer_asset_mods(self) -> None:
        """Re-run asset remaps after a renderer hot-swap (native downscaling)."""
        if self._has_change_background:
            self._apply_change_background_color()

    def _apply_change_background_color(self):
        """Remap baked background IDs to magenta; preserve the left black pillar."""
        import numpy as np

        renderer = self._env.renderer
        new_rgb = tuple(ChangeBackgroundColorMod._NEW_BG)
        if new_rgb not in renderer.COLOR_TO_ID:
            raise KeyError(
                f"change_background_color: palette missing swatch {new_rgb}; "
                "ChangeBackgroundColorMod.asset_overrides must seed it."
            )
        new_id = int(renderer.COLOR_TO_ID[new_rgb])
        black_id = renderer.COLOR_TO_ID.get((0, 0, 0))
        frames = np.asarray(renderer.BACKGROUND_FRAMES)
        if black_id is None:
            frames = np.full_like(frames, new_id)
        else:
            frames = np.where(frames != int(black_id), new_id, frames)
        renderer.BACKGROUND_FRAMES = jnp.asarray(frames)
        renderer.BACKGROUND = renderer.BACKGROUND_FRAMES[0]
        alt_i = 1 if renderer.BACKGROUND_FRAMES.shape[0] > 1 else 0
        renderer.BACKGROUND_ALT = renderer.BACKGROUND_FRAMES[alt_i]

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

            # Collected divers blink when there are 6 of them
            visible_divers = jax.lax.select(
                jnp.logical_and(state.divers_collected == 6, (state.step_counter % 8) >= 4),
                0,
                state.divers_collected
            )
            raster = jr.render_indicator(raster, 49, 178, visible_divers, renderer.SHAPE_MASKS['diver_indicator'], spacing=10, max_value=6)

            raster = jr.render_bar(raster, 49, 170, state.oxygen, 64, 63, 5, renderer.OXYGEN_COLOR_ID, renderer.OXYGEN_BAR_BG_COLOR_ID)

            raster = jr.draw_rects(
                raster,
                positions=jnp.array([[0, 0]]),
                sizes=jnp.array([[8, renderer.config.game_dimensions[0]]]),
                color_id=renderer.BACKGROUND[0, 0]
            )

            return jr.render_from_palette(raster, renderer.PALETTE)

        return self._env.render(state)
