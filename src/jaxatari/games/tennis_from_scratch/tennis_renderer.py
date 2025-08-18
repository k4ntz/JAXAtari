import jax
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj
import os
import chex

from tennis_main import FRAME_WIDTH, FRAME_HEIGHT, GAME_OFFSET_LEFT_BOTTOM, GAME_OFFSET_TOP, GAME_HEIGHT, GAME_WIDTH, \
    TennisState, PLAYER_WIDTH, PLAYER_HEIGHT


def recolor_blue_to_red(sprite, blue_color=[117, 128, 240, 255], red_color=[240, 128, 128, 255]):
    # Convert color constants to jax arrays of the same dtype
    blue_color = jnp.array(blue_color, dtype=sprite.dtype)
    red_color = jnp.array(red_color, dtype=sprite.dtype)

    # Create a mask: shape (H, W), where each pixel matches the blue color
    mask = jnp.all(sprite == blue_color, axis=-1)  # shape: (H, W)

    # Find the indices of the pixels to replace
    indices = jnp.argwhere(mask)  # shape: (N, 2)

    # Replace each matching pixel using .at[].set()
    for idx in indices:
        sprite = sprite.at[tuple(idx)].set(red_color)

    return sprite


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
    BG = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/background.npy")), axis=0)
    BALL = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ball.npy")), axis=0)
    BALL_SHADOW = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ball_shadow.npy")),
                                  axis=0)
    PLAYER_0 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/player_no_racket.npy")), axis=0)
    PLAYER_1 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/player_no_racket_1.npy")), axis=0)
    PLAYER_2 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/player_no_racket_2.npy")), axis=0)
    PLAYER_3 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/player_no_racket_3.npy")), axis=0)
    RACKET_0 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/1.npy")), axis=0)
    RACKET_1 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/2.npy")), axis=0)
    RACKET_2 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/3.npy")), axis=0)
    RACKET_3 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/4.npy")), axis=0)

    # UI sprites
    UI_NUM_0 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_0.npy")),
                               axis=0)
    UI_NUM_1 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_1.npy")),
                               axis=0)
    UI_NUM_2 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_2.npy")),
                               axis=0)
    UI_NUM_3 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_3.npy")),
                               axis=0)
    UI_NUM_4 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_4.npy")),
                               axis=0)
    UI_NUM_5 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_5.npy")),
                               axis=0)
    UI_NUM_6 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_6.npy")),
                               axis=0)
    UI_NUM_7 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_7.npy")),
                               axis=0)
    UI_NUM_8 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_8.npy")),
                               axis=0)
    UI_NUM_9 = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_9.npy")),
                               axis=0)
    UI_DEUCE = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_9.npy")),
                               axis=0)
    UI_AD_IN = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_9.npy")),
                               axis=0)
    UI_AD_OUT = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ui_blue_9.npy")),
                                axis=0)

    return (BG, switch_blue_and_red(BG), BALL, BALL_SHADOW,
            [switch_blue_and_red(PLAYER_0), switch_blue_and_red(PLAYER_1), switch_blue_and_red(PLAYER_2),
             switch_blue_and_red(PLAYER_3)], [PLAYER_0, PLAYER_1, PLAYER_2, PLAYER_3],
            [switch_blue_and_red(RACKET_3), switch_blue_and_red(RACKET_0), switch_blue_and_red(RACKET_1),
             switch_blue_and_red(RACKET_2)],
            [RACKET_3, RACKET_0, RACKET_1, RACKET_2],
            [UI_NUM_0, UI_NUM_1, UI_NUM_2, UI_NUM_3, UI_NUM_4, UI_NUM_5, UI_NUM_6, UI_NUM_7, UI_NUM_8, UI_NUM_9],
            [switch_blue_and_red(UI_NUM_0), switch_blue_and_red(UI_NUM_1), switch_blue_and_red(UI_NUM_2),
             switch_blue_and_red(UI_NUM_3), switch_blue_and_red(UI_NUM_4), switch_blue_and_red(UI_NUM_5),
             switch_blue_and_red(UI_NUM_6), switch_blue_and_red(UI_NUM_7), switch_blue_and_red(UI_NUM_8),
             switch_blue_and_red(UI_NUM_9)],
            UI_DEUCE,
            UI_AD_IN,
            UI_AD_OUT)


# ToDo remove
def get_bounding_box(width, height, border_color=(255, 0, 0, 255), border_thickness=1):
    """
    Creates a bounding box sprite with transparent interior and colored border.

    Args:
        width: Width of the bounding box (static)
        height: Height of the bounding box (static)
        border_color: RGBA tuple for border color (default: red)
        border_thickness: Pixel width of the border (default: 1)

    Returns:
        Array of shape (height, width, 4) with RGBA values
    """
    # Create empty RGBA array
    sprite = jnp.zeros((width, height, 4), dtype=jnp.uint8)

    # Create border mask
    border_mask = jnp.zeros((width, height), dtype=bool)

    # Set border pixels to True
    border_mask = border_mask.at[:border_thickness, :].set(True)  # Top border
    border_mask = border_mask.at[-border_thickness:, :].set(True)  # Bottom border
    border_mask = border_mask.at[:, :border_thickness].set(True)  # Left border
    border_mask = border_mask.at[:, -border_thickness:].set(True)  # Right border

    # Apply border color where mask is True
    sprite = jnp.where(
        border_mask[..., None],  # Expand for RGBA channels
        jnp.array(border_color, dtype=jnp.uint8),
        sprite
    )

    return sprite[jnp.newaxis, ...]


def perspective_transform(x, y, apply_offsets=True, width_top=79.0, width_bottom=111.0, height=130.0):
    # Normalize y: 0 at top (far), 1 at bottom (near)
    y_norm = y / height

    # Interpolate width at this y level
    current_width = width_top * (1 - y_norm) + width_bottom * y_norm

    # Horizontal offset to center the perspective slice
    offset = (width_bottom - current_width) / 2
    # offset = 0

    # Normalize x based on bottom width (field space)
    x_norm = x / width_bottom

    # Compute final x position
    x_screen = offset + x_norm * current_width
    y_screen = y  # No vertical scaling

    if apply_offsets:
        return x_screen + GAME_OFFSET_LEFT_BOTTOM, y_screen + GAME_OFFSET_TOP
    return x_screen, y_screen


class TennisRenderer:

    def __init__(self):
        (self.BG_TOP_RED, self.BG_TOP_BLUE, self.BALL, self.BALL_SHADOW, self.PLAYER_BLUE, self.PLAYER_RED,
         self.RACKET_BLUE, self.RACKET_RED, self.UI_NUMBERS_BLUE,
         self.UI_NUMBERS_RED, self.UI_DEUCE, self.UI_AD_IN, self.UI_AD_OUT) = load_sprites()
        # use bounding box as mockup
        self.BOUNDING_BOX = get_bounding_box(PLAYER_WIDTH, PLAYER_HEIGHT)
        self.PLAYER_X_BOX = get_bounding_box(1, 3, border_color=(0, 0, 0, 255))
        self.ENEMY_X_BOX = get_bounding_box(1, 3, border_color=(0, 0, 0, 255))
        self.ENEMY_BOX = get_bounding_box(PLAYER_WIDTH, PLAYER_HEIGHT)

    def render_number_centered(self, raster, number: int, position, red=False):
        chars = list(str(number))

        all_sprites = self.UI_NUMBERS_RED if red else self.UI_NUMBERS_BLUE
        sprites = [aj.get_sprite_frame((all_sprites[0] if int(c) >= len(all_sprites) else all_sprites[int(c)]), 0) for c
                   in chars]

        padding = 2
        total_width = sum(s.shape[0] for s in sprites) + (len(sprites) - 1) * padding

        curr_x = position[0] - total_width / 2
        for sprite in sprites:
            raster = aj.render_at(raster, curr_x, position[1],
                                  sprite)

            curr_x += sprite.shape[0] + padding

        return raster

    # @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TennisState) -> jnp.ndarray:
        raster = jnp.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3))

        frame_bg = aj.get_sprite_frame(self.BG_TOP_RED if state.player_state.player_field == 1 else self.BG_TOP_BLUE, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_ball_shadow = aj.get_sprite_frame(self.BALL_SHADOW, 0)
        # calculate screen coordinates of ball
        # ball_screen_x, ball_screen_y = self.perspective_transform(state.ball_state.ball_x, state.ball_state.ball_y)
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y, frame_ball_shadow)

        frame_ball = aj.get_sprite_frame(self.BALL, 0)
        # apply flat y offset depending on z value
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y - state.ball_state.ball_z,
                              frame_ball)

        # print("x ball: {:0.2f}., y ball: {:0.2f}, z ball: {:0.2f}, vel: {:0.2f}".format(state.ball_state.ball_x, state.ball_state.ball_y, state.ball_state.ball_z, state.ball_state.ball_velocity_z_fp))
        print(f"x ball: {state.ball_state.ball_x :0.2f} y ball: {state.ball_state.ball_y :0.2f} x enemy: {state.enemy_state.enemy_x :0.2f} y enemy: {state.enemy_state.enemy_y :0.2f}")

        frame_player = aj.get_sprite_frame(self.PLAYER_RED[state.animator_state.player_frame], 0)
        if state.player_state.player_direction == -1:
            frame_player = jnp.flip(frame_player, axis=0)

        frame_enemy = aj.get_sprite_frame(self.PLAYER_BLUE[state.animator_state.enemy_frame], 0)
        if state.enemy_state.enemy_direction == -1:
            frame_enemy = jnp.flip(frame_enemy, axis=0)

        player_pos = state.player_state.player_x - 2 if state.player_state.player_direction == 1 else state.player_state.player_x - 4

        racket_offset_x = [0, 1, 4, 3]
        player_racket_pos = state.player_state.player_x - 2 if state.player_state.player_direction == 1 else state.player_state.player_x - \
                                                                                                             racket_offset_x[
                                                                                                                 state.animator_state.player_racket_frame]
        raster = aj.render_at(raster, player_pos, state.player_state.player_y, frame_player)

        racket_offset = [1, 8, 8, 4]
        frame_racket_player = aj.get_sprite_frame(self.RACKET_RED[state.animator_state.player_racket_frame], 0)
        if state.player_state.player_direction == -1:
            frame_racket_player = jnp.flip(frame_racket_player, axis=0)
        raster = aj.render_at(raster, player_racket_pos + state.player_state.player_direction * 8,
                              state.player_state.player_y + racket_offset[state.animator_state.player_racket_frame],
                              frame_racket_player)



        frame_bounding_box = aj.get_sprite_frame(self.BOUNDING_BOX, 0)
        frame_enemy_box = aj.get_sprite_frame(self.ENEMY_BOX, 0)
        enemy_box_pos = state.enemy_state.enemy_x if state.enemy_state.enemy_direction == 1 else state.enemy_state.enemy_x - PLAYER_WIDTH + 2
        raster = aj.render_at(raster, enemy_box_pos, state.enemy_state.enemy_y, frame_enemy_box)
        bounding_box_pos = state.player_state.player_x if state.player_state.player_direction == 1 else state.player_state.player_x - PLAYER_WIDTH + 2
        raster = aj.render_at(raster, bounding_box_pos, state.player_state.player_y, frame_bounding_box)


        enemy_pos = state.enemy_state.enemy_x - 2 if state.enemy_state.enemy_direction == 1 else state.enemy_state.enemy_x - 4
        enemy_racket_pos = state.enemy_state.enemy_x - 2 if state.enemy_state.enemy_direction == 1 else state.enemy_state.enemy_x - \
                                                                                                        racket_offset_x[
                                                                                                            state.animator_state.enemy_racket_frame]

        frame_racket_enemy = aj.get_sprite_frame(self.RACKET_BLUE[state.animator_state.enemy_racket_frame], 0)
        if state.enemy_state.enemy_direction == -1:
            frame_racket_enemy = jnp.flip(frame_racket_enemy, axis=0)
        raster = aj.render_at(raster, enemy_pos, state.enemy_state.enemy_y, frame_enemy)
        raster = aj.render_at(raster, enemy_racket_pos + state.enemy_state.enemy_direction * 8,
                              state.enemy_state.enemy_y + racket_offset[state.animator_state.enemy_racket_frame],
                              frame_racket_enemy)

        player_x_rec = jnp.zeros((2, 2, 4))
        player_x_rec = player_x_rec.at[:, :, 1].set(255)  # Yellow
        player_x_rec = player_x_rec.at[:, :, 2].set(255)  # Yellow
        player_x_rec = player_x_rec.at[:, :, 3].set(255)  # Alpha

        raster = aj.render_at(raster, state.player_state.player_x, state.player_state.player_y, player_x_rec)

        enemy_x_rec = jnp.zeros((2, 2, 4))
        enemy_x_rec = enemy_x_rec.at[:, :, 1].set(255)  # Yellow
        enemy_x_rec = enemy_x_rec.at[:, :, 2].set(255)  # Yellow
        enemy_x_rec = enemy_x_rec.at[:, :, 3].set(255)  # Alpha

        raster = aj.render_at(raster, state.enemy_state.enemy_x, state.enemy_state.enemy_y, enemy_x_rec)

        top_left_rec = jnp.zeros((2, 2, 4))
        top_left_rec = top_left_rec.at[:, :, 1].set(255)  # Yellow
        top_left_rec = top_left_rec.at[:, :, 0].set(255)  # Yellow
        top_left_rec = top_left_rec.at[:, :, 3].set(255)  # Alpha
        top_left_corner_coords = perspective_transform(0, 0)

        raster = aj.render_at(raster, top_left_corner_coords[0], top_left_corner_coords[1], top_left_rec)

        top_right_rec = jnp.zeros((2, 2, 4))
        top_right_rec = top_right_rec.at[:, :, 2].set(255)  # Blue
        top_right_rec = top_right_rec.at[:, :, 3].set(255)  # Alpha
        top_right_corner_coords = perspective_transform(GAME_WIDTH, 0)

        raster = aj.render_at(raster, top_right_corner_coords[0], top_right_corner_coords[1], top_right_rec)

        bottom_left_rec = jnp.zeros((2, 2, 4))
        bottom_left_rec = bottom_left_rec.at[:, :, 1].set(255)  # Green
        bottom_left_rec = bottom_left_rec.at[:, :, 3].set(255)  # Alpha
        bottom_left_corner_coords = perspective_transform(0, GAME_HEIGHT)

        raster = aj.render_at(raster, bottom_left_corner_coords[0], bottom_left_corner_coords[1], bottom_left_rec)

        bottom_right_rec = jnp.zeros((2, 2, 4))
        bottom_right_rec = bottom_right_rec.at[:, :, 0].set(255)  # Red
        bottom_right_rec = bottom_right_rec.at[:, :, 3].set(255)  # Alpha
        bottom_right_corner_coords = perspective_transform(GAME_WIDTH, GAME_HEIGHT)

        raster = aj.render_at(raster, bottom_right_corner_coords[0], bottom_right_corner_coords[1], bottom_right_rec)

        if (
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
                raster = self.render_number_centered(raster, enemy_score_number, [(FRAME_WIDTH / 4) * 3, 2])

        return raster
