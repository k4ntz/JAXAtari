import os
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pong.pong_mod_plugins import LazyEnemyMod, RandomEnemyMod, AlwaysZeroScoreMod, LinearMovementMod, ShiftPlayerMod, ShiftEnemyMod, NoFireMod, ChangeBackgroundColorMod, ChangePlayerColorMod, TriplePongMod

class PongEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Pong.
    It simply inherits all logic from JaxAtariModController and defines the PONG_MOD_REGISTRY.
    """

    REGISTRY = {
        "lazy_enemy": LazyEnemyMod,
        "random_enemy": RandomEnemyMod,
        "zero_score": AlwaysZeroScoreMod,
        "linear_movement": LinearMovementMod,
        "shift_player": ShiftPlayerMod,
        "shift_enemy": ShiftEnemyMod,
        "no_fire": NoFireMod,
        "change_background_color": ChangeBackgroundColorMod,
        "change_player_color": ChangePlayerColorMod,
        "triple_pong": TriplePongMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "pong", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        self._has_triple_pong = "triple_pong" in mods_config
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY  # for pong this is the only specific part, but other games might need to do execute some other logic in the constructor.
        )

    @partial(jax.jit, static_argnames=['self'])
    def render(self, state):
        """
        Render hook. With the 'triple_pong' mod active this draws the 3 balls
        (each in its own color, not-yet-spawned ones hidden) and the 3 enemy
        paddles; otherwise it simply delegates to the base environment.
        """
        if not self._has_triple_pong:
            return self._env.render(state)

        renderer = self._env.renderer
        jr = renderer.jr
        consts = self._env.consts
        n = TriplePongMod.NUM_BALLS

        raster = jr.create_object_raster(renderer.BACKGROUND)

        # Player paddle (unchanged, single)
        raster = jr.render_at(
            raster,
            consts.PLAYER_X,
            jnp.round(state.player_y).astype(jnp.int32),
            renderer.SHAPE_MASKS["player"],
        )

        # Enemy paddles: 3 paddles at ENEMY_X, each restricted to its zone
        enemy_masks = jnp.tile(renderer.SHAPE_MASKS["enemy"], (n, 1, 1))
        enemy_x = jnp.full((n,), consts.ENEMY_X, dtype=jnp.int32)
        raster = jr.render_at_batch(raster, enemy_x, state.enemy_y, enemy_masks)

        # Balls: distinct colors per ball; balls that have not spawned yet are
        # drawn fully transparent so they stay hidden at the center.
        ball_masks = jnp.stack([
            renderer.SHAPE_MASKS["ball"],
            renderer.SHAPE_MASKS["ball_red"],
            renderer.SHAPE_MASKS["ball_blue"],
        ])
        active = state.ball_vel_x != 0
        transparent = jnp.full_like(renderer.SHAPE_MASKS["ball"], jr.TRANSPARENT_ID)
        visible_masks = jnp.where(active[:, None, None], ball_masks, transparent[None, :, :])
        raster = jr.render_at_batch(raster, state.ball_x, state.ball_y, visible_masks)

        # --- Walls ---
        raster = jr.render_at(raster, 0, consts.WALL_TOP_Y, renderer.SHAPE_MASKS["wall_top"])
        raster = jr.render_at(raster, 0, consts.WALL_BOTTOM_Y, renderer.SHAPE_MASKS["wall_bottom"])

        # --- Scores ---
        player_digits = jr.int_to_digits(state.player_score, max_digits=2)
        enemy_digits = jr.int_to_digits(state.enemy_score, max_digits=2)

        player_digit_masks = renderer.SHAPE_MASKS["player_digits"]
        enemy_digit_masks = renderer.SHAPE_MASKS["enemy_digits"]

        is_player_single_digit = state.player_score < 10
        player_start_index = jax.lax.select(is_player_single_digit, 1, 0)
        player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)
        player_render_x = jax.lax.select(is_player_single_digit,
                                         120 + 16 // 2,
                                         120)

        raster = jr.render_label_selective(raster, player_render_x, 3, player_digits, player_digit_masks, player_start_index, player_num_to_render, spacing=16)

        is_enemy_single_digit = state.enemy_score < 10
        enemy_start_index = jax.lax.select(is_enemy_single_digit, 1, 0)
        enemy_num_to_render = jax.lax.select(is_enemy_single_digit, 1, 2)
        enemy_render_x = jax.lax.select(is_enemy_single_digit,
                                        10 + 16 // 2,
                                        10)

        raster = jr.render_label_selective(raster, enemy_render_x, 3, enemy_digits, enemy_digit_masks, enemy_start_index, enemy_num_to_render, spacing=16)

        return jr.render_from_palette(raster, renderer.PALETTE)
