import jax
import jax.numpy as jnp
import jaxatari.rendering.jax_rendering_utils as aj
import os
import chex

from tennis_main import FRAME_WIDTH, FRAME_HEIGHT, TennisState


def switch_blue_and_red(sprite, blue_color=[117, 128, 240, 255], red_color=[240, 128, 128, 255]):
    # Convert color constants to jax arrays of the same dtype
    blue_color = jnp.array(blue_color, dtype=sprite.dtype)
    red_color = jnp.array(red_color, dtype=sprite.dtype)

    # Create a mask: shape (H, W), where each pixel matches the blue color
    mask_blue = jnp.all(sprite == blue_color, axis=-1)  # shape: (H, W)
    mask_red = jnp.all(sprite == red_color, axis=-1)  # shape: (H, W)

    # Find the indices of the pixels to replace
    indices_blue = jnp.argwhere(mask_blue)  # shape: (N, 2)
    indices_red = jnp.argwhere(mask_red)  # shape: (N, 2)

    # Replace each matching pixel using .at[].set()
    for idx in indices_blue:
        sprite = sprite.at[tuple(idx)].set(red_color)
    for idx in indices_red:
        sprite = sprite.at[tuple(idx)].set(blue_color)

    return sprite


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BG = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/background.npy")), axis=0)
    BALL = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ball.npy")), axis=0)
    BALL_SHADOW = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ball_shadow.npy")),
                                  axis=0)
    PLAYER_0 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket.npy")), axis=0)
    PLAYER_1 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket_1.npy")), axis=0)
    PLAYER_2 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket_2.npy")), axis=0)
    PLAYER_3 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket_3.npy")), axis=0)
    UI_DEUCE = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_9.npy")),
                               axis=0)
    UI_AD_IN = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_9.npy")),
                               axis=0)
    UI_AD_OUT = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_9.npy")),
                                axis=0)

    def recolor_rgba(stack, src_rgba, dst_rgba):
        # stack shape: (10, 1, H, W, 4) or (10, H, W, 4); dtype uint8
        src = jnp.asarray(src_rgba, dtype=stack.dtype)
        dst = jnp.asarray(dst_rgba, dtype=stack.dtype)
        mask = jnp.all(stack == src, axis=-1, keepdims=True)  # shape (...,1)
        return jnp.where(mask, dst, stack)

    RACKETS_RED = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "games/sprites/tennis/racket_red_{}.npy"), num_chars=4
    )
    RACKETS_BLUE = recolor_rgba(RACKETS_RED, [240, 128, 128, 255], [117, 128, 240, 255])

    UI_NUMBERS_BLUE = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_{}.npy"), num_chars=10
    )
    UI_NUMBERS_RED = recolor_rgba(UI_NUMBERS_BLUE, [117, 128, 240, 255], [240, 128, 128, 255])

    return (BG, switch_blue_and_red(BG), BALL, BALL_SHADOW,
            jnp.asarray([switch_blue_and_red(PLAYER_0), switch_blue_and_red(PLAYER_1), switch_blue_and_red(PLAYER_2),
                         switch_blue_and_red(PLAYER_3)]), jnp.asarray([PLAYER_0, PLAYER_1, PLAYER_2, PLAYER_3]),
            RACKETS_BLUE,
            RACKETS_RED,
            UI_NUMBERS_BLUE,
            UI_NUMBERS_RED,
            UI_DEUCE,
            UI_AD_IN,
            UI_AD_OUT)


class TennisRenderer:

    def __init__(self):
        (self.BG_TOP_RED, self.BG_TOP_BLUE, self.BALL, self.BALL_SHADOW, self.PLAYER_BLUE, self.PLAYER_RED,
         self.RACKET_BLUE, self.RACKET_RED, self.UI_NUMBERS_BLUE,
         self.UI_NUMBERS_RED, self.UI_DEUCE, self.UI_AD_IN, self.UI_AD_OUT) = load_sprites()

    def render_number_centered(self, raster, number, position, red=False):
        digits = aj.int_to_digits(number, max_digits=2)  # or 2
        sprites = self.UI_NUMBERS_RED if red else self.UI_NUMBERS_BLUE

        n = jnp.asarray(number, jnp.int32)

        # is the value 1 digit?
        is_single = n < 10

        # For max_digits = 2:
        start_idx = jax.lax.select(is_single, 1, 0)  # skip leading zero when single digit
        count = jax.lax.select(is_single, 1, 2)

        raster = aj.render_label_selective(
            raster, position[0], position[1],
            digits, sprites,
            start_idx, count,
            spacing=7
        )

        return raster

    def render(self, state: TennisState) -> jnp.ndarray:
        raster = aj.create_initial_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT)

        # render background
        bg_top_red = jnp.where(state.player_state.player_field == 1, True, False)

        raster = jax.lax.cond(
            bg_top_red,
            lambda r: aj.render_at(
                r,
                0,
                0,
                aj.get_sprite_frame(self.BG_TOP_RED, 0),
            ),
            lambda r: aj.render_at(
                r,
                0,
                0,
                aj.get_sprite_frame(self.BG_TOP_BLUE, 0),
            ),
            raster,
        )

        # render ball
        frame_ball_shadow = aj.get_sprite_frame(self.BALL_SHADOW, 0)
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y, frame_ball_shadow)

        frame_ball = aj.get_sprite_frame(self.BALL, 0)
        # apply flat y offset depending on z value
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y - state.ball_state.ball_z,
                              frame_ball)

        # render player & enemy
        frame_player = aj.get_sprite_frame(self.PLAYER_RED[state.animator_state.player_frame], 0)
        frame_enemy = aj.get_sprite_frame(self.PLAYER_BLUE[state.animator_state.enemy_frame], 0)

        player_pos = jnp.where(state.player_state.player_direction == 1,
                               state.player_state.player_x - 2,
                               state.player_state.player_x - 4)

        racket_offset_x = jnp.asarray([0, 1, 2, 2])
        player_racket_pos = jnp.where(state.player_state.player_direction == 1,
                                      state.player_state.player_x - 4 + racket_offset_x[
                                          state.animator_state.player_racket_frame],
                                      state.player_state.player_x - racket_offset_x[
                                          state.animator_state.player_racket_frame] - 2)

        raster = aj.render_at(raster, player_pos, state.player_state.player_y, frame_player,
                              flip_horizontal=jnp.where(state.player_state.player_direction == -1, True, False))

        racket_offset = jnp.asarray([1, 8, 8, 4])

        frame_racket_player = aj.get_sprite_frame(self.RACKET_RED, state.animator_state.player_racket_frame)

        raster = aj.render_at(raster, player_racket_pos + state.player_state.player_direction * 8,
                              state.player_state.player_y + racket_offset[state.animator_state.player_racket_frame],
                              frame_racket_player,
                              flip_horizontal=jnp.where(state.player_state.player_direction == -1, True, False))

        enemy_pos = jnp.where(state.enemy_state.enemy_direction == 1,
                              state.enemy_state.enemy_x - 2,
                              state.enemy_state.enemy_x - 4)
        enemy_racket_pos = jnp.where(state.enemy_state.enemy_direction == 1,
                                     state.enemy_state.enemy_x - 4 + racket_offset_x[
                                         state.animator_state.enemy_racket_frame],
                                     state.enemy_state.enemy_x - racket_offset_x[
                                         state.animator_state.enemy_racket_frame] - 2)

        raster = aj.render_at(raster, enemy_pos, state.enemy_state.enemy_y, frame_enemy,
                              flip_horizontal=jnp.where(state.enemy_state.enemy_direction == -1, True, False))

        frame_racket_enemy = aj.get_sprite_frame(self.RACKET_BLUE, state.animator_state.enemy_racket_frame)

        raster = aj.render_at(raster, enemy_racket_pos + state.enemy_state.enemy_direction * 8,
                              state.enemy_state.enemy_y + racket_offset[state.animator_state.enemy_racket_frame],
                              frame_racket_enemy,
                              flip_horizontal=jnp.where(state.enemy_state.enemy_direction == -1, True, False))

        # render score UI
        should_display_overall_score = jnp.logical_and((
                                                               state.game_state.player_game_score + state.game_state.enemy_game_score) > 0,
                                                       jnp.logical_and(state.game_state.player_score == 0,
                                                                       state.game_state.enemy_score == 0))

        def render_overall(r):
            # display overall score
            r = self.render_number_centered(r, state.game_state.player_game_score, [FRAME_WIDTH / 4, 2],
                                            red=True)
            r = self.render_number_centered(r, state.game_state.enemy_game_score, [(FRAME_WIDTH / 4) * 3, 2])
            return r

        def render_current_score(raster):
            tennis_scores = jnp.array([0, 15, 30, 40], dtype=jnp.int32)

            ps = state.game_state.player_score
            es = state.game_state.enemy_score
            serving = state.player_state.player_serving  # bool

            # "deuce-like" phase once both reached 40 (i.e., >= 3 points)
            deuce_like = (ps >= 3) & (es >= 3)

            def render_deuce(raster_in):
                # deuce if tied; otherwise advantage
                is_tied = (ps == es)

                def do_deuce(r):
                    ui = aj.get_sprite_frame(self.UI_DEUCE, 0)
                    x = (FRAME_WIDTH // 4) - (ui.shape[0] // 2)
                    return aj.render_at(r, x, 2, ui)

                def do_adv(r):
                    # AD-IN if leader is the server; otherwise AD-OUT
                    ad_in = ((ps > es) & serving) | ((es > ps) & (~serving))

                    def render_ad_in(rr):
                        ui = aj.get_sprite_frame(self.UI_AD_IN, 0)
                        x = (FRAME_WIDTH // 4) - (ui.shape[0] // 2)
                        return aj.render_at(rr, x, 2, ui)

                    def render_ad_out(rr):
                        ui = aj.get_sprite_frame(self.UI_AD_OUT, 0)
                        x = (FRAME_WIDTH // 4) - (ui.shape[0] // 2)
                        return aj.render_at(rr, x, 2, ui)

                    return jax.lax.cond(ad_in, render_ad_in, render_ad_out, r)

                return jax.lax.cond(is_tied, do_deuce, do_adv, raster_in)

            def render_regular(raster_in):
                # clip score indices to [0..3] and map to 0/15/30/40
                pid = jnp.minimum(3, ps)
                eid = jnp.minimum(3, es)
                pnum = tennis_scores[pid]
                enum = tennis_scores[eid]

                r = self.render_number_centered(raster_in, pnum, [FRAME_WIDTH // 4, 2], red=True)
                r = self.render_number_centered(r, enum, [(FRAME_WIDTH // 4) * 3, 2])
                return r

            raster = jax.lax.cond(deuce_like, render_deuce, render_regular, raster)
            return raster

        raster = jax.lax.cond(
            should_display_overall_score,
            lambda r: render_overall(r),
            lambda r: render_current_score(r),
            raster,
        )

        """if (
                state.game_state.player_game_score + state.game_state.enemy_game_score) > 0 and state.game_state.player_score == 0 and state.game_state.enemy_score == 0:
            # display overall score
            raster = self.render_number_centered(raster, state.game_state.player_game_score, [FRAME_WIDTH / 4, 2],
                                                 red=True)
            raster = self.render_number_centered(raster, state.game_state.enemy_game_score, [(FRAME_WIDTH / 4) * 3, 2])

        else:
            # display current set score
            tennis_scores = [0, 15, 30, 40]

            # deuce situation
            if (state.game_state.player_score >= len(tennis_scores) or state.game_state.enemy_score >= len(
                    tennis_scores)) or (
                    state.game_state.player_score >= len(tennis_scores) - 1 and state.game_state.enemy_score >= len(
                    tennis_scores) - 1):
                if state.game_state.player_score == state.game_state.enemy_score:
                    ui_deuce_sprite = aj.get_sprite_frame(self.UI_DEUCE, 0)
                    raster = aj.render_at(raster, FRAME_WIDTH / 4 - ui_deuce_sprite.shape[0] / 2, 2,
                                          ui_deuce_sprite)
                elif (
                        state.game_state.player_score > state.game_state.enemy_score and state.player_state.player_serving) or (
                        state.game_state.enemy_score > state.game_state.player_score and not state.player_state.player_serving):
                    ui_ad_in_sprite = aj.get_sprite_frame(self.UI_AD_IN, 0)
                    raster = aj.render_at(raster, FRAME_WIDTH / 4 - ui_ad_in_sprite.shape[0] / 2, 2,
                                          ui_ad_in_sprite)
                else:
                    ui_ad_out_sprite = aj.get_sprite_frame(self.UI_AD_OUT, 0)
                    raster = aj.render_at(raster, FRAME_WIDTH / 4 - ui_ad_out_sprite.shape[0] / 2, 2,
                                          ui_ad_out_sprite)
            # regular play
            else:
                player_score_number = tennis_scores[min(len(tennis_scores) - 1, state.game_state.player_score)]
                enemy_score_number = tennis_scores[min(len(tennis_scores) - 1, state.game_state.enemy_score)]

                raster = self.render_number_centered(raster, player_score_number, [FRAME_WIDTH / 4, 2], red=True)
                raster = self.render_number_centered(raster, enemy_score_number, [(FRAME_WIDTH / 4) * 3, 2])"""

        return raster
