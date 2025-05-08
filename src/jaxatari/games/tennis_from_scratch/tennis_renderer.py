from functools import partial
import jax
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj
import os

from tennis_main import FRAME_WIDTH, FRAME_HEIGHT, GAME_OFFSET_LEFT_BOTTOM, GAME_OFFSET_TOP, GAME_HEIGHT, GAME_WIDTH, TennisState

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BG = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/background.npy")), axis=0)
    BALL = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ball.npy")), axis=0)
    BALL_SHADOW = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/ball_shadow.npy")), axis=0)

    return BG, BALL, BALL_SHADOW

class TennisRenderer:

    def __init__(self):
        (self.BG, self.BALL, self.BALL_SHADOW) = load_sprites()

    #@partial(jax.jit, static_argnums=(0,))
    def render(self, state: TennisState):
        raster = jnp.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3))

        frame_bg = aj.get_sprite_frame(self.BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_ball = aj.get_sprite_frame(self.BALL, 0)
        # calculate screen coordinates of ball
        screen_x, screen_y = self.perspective_transform2(state.ball_x, state.ball_y)
        # calculate screen offset from left of frame
        screen_offset_x, _ = self.perspective_transform2(GAME_OFFSET_LEFT_BOTTOM, state.ball_y)

        """for i in range(0, GAME_HEIGHT):
            rec_left, _ = self.perspective_transform2(0, i)
            rec_right, _ = self.perspective_transform2(GAME_WIDTH, i)
            rec_width = rec_right - rec_left
            rectangle = jnp.zeros((int(rec_width), 1, 4))
            rectangle = rectangle.at[:, :, 0].set(255)  # Red
            rectangle = rectangle.at[:, :, 3].set(255)  # Alpha

            raster = aj.render_at(raster, i + GAME_OFFSET_TOP, rec_left + 2, rectangle)"""

        #raster = aj.render_at(raster, screen_x, screen_y, frame_ball)
        raster = aj.render_at(raster, screen_y, screen_x, frame_ball)

        top_left_rec = jnp.zeros((2, 2, 4))
        top_left_rec = top_left_rec.at[:, :, 1].set(255)  # Yellow
        top_left_rec = top_left_rec.at[:, :, 0].set(255)  # Yellow
        top_left_rec = top_left_rec.at[:, :, 3].set(255)  # Alpha
        top_left_corner_coords = self.perspective_transform2(0, 0)

        raster = aj.render_at(raster, top_left_corner_coords[1], top_left_corner_coords[0], top_left_rec)

        top_right_rec = jnp.zeros((2, 2, 4))
        top_right_rec = top_right_rec.at[:, :, 2].set(255)  # Blue
        top_right_rec = top_right_rec.at[:, :, 3].set(255)  # Alpha
        top_right_corner_coords = self.perspective_transform2(GAME_WIDTH, 0)

        raster = aj.render_at(raster, top_right_corner_coords[1], top_right_corner_coords[0], top_right_rec)

        bottom_left_rec = jnp.zeros((2, 2, 4))
        bottom_left_rec = bottom_left_rec.at[:, :, 1].set(255)  # Green
        bottom_left_rec = bottom_left_rec.at[:, :, 3].set(255)  # Alpha
        bottom_left_corner_coords = self.perspective_transform2(0, GAME_HEIGHT)

        raster = aj.render_at(raster, bottom_left_corner_coords[1], bottom_left_corner_coords[0], bottom_left_rec)

        bottom_right_rec = jnp.zeros((2, 2, 4))
        bottom_right_rec = bottom_right_rec.at[:, :, 0].set(255)  # Red
        bottom_right_rec = bottom_right_rec.at[:, :, 3].set(255)  # Alpha
        bottom_right_corner_coords = self.perspective_transform2(GAME_WIDTH, GAME_HEIGHT)

        raster = aj.render_at(raster, bottom_right_corner_coords[1], bottom_right_corner_coords[0], bottom_right_rec)

        #raster = aj.render_at(raster, 0, 0, rectangle)

        return raster

    # we always use coordinates including the lines

    def perspective_transform(self, x, y, width_top = 79.0, width_bottom = 111.0, height = 130.0):
        normalized_y = y / height

        # interpolate between bottom and top width to get current width
        current_width = width_bottom * (1 - normalized_y) + width_top * normalized_y

        # assume width_bottom is the "unskewed" width, meaning x needs to be scaled down for width_top
        normalized_x = x / width_bottom

        screen_x = (1 - current_width) / 2 + normalized_x * current_width

        return screen_x * width_bottom, y

    def perspective_transform2(self, x, y, apply_offsets = True, width_top = 79.0, width_bottom = 111.0, height = 130.0):
        # Normalize y: 0 at top (far), 1 at bottom (near)
        y_norm = y / height

        # Interpolate width at this y level
        current_width = width_top * (1 - y_norm) + width_bottom * y_norm

        # Horizontal offset to center the perspective slice
        offset = (width_bottom - current_width) / 2
        #offset = 0

        # Normalize x based on bottom width (field space)
        x_norm = x / width_bottom

        # Compute final x position
        x_screen = offset + x_norm * current_width
        y_screen = y  # No vertical scaling (unless you want depth stretching)

        if apply_offsets:
            return x_screen + GAME_OFFSET_LEFT_BOTTOM, y_screen + GAME_OFFSET_TOP
        return x_screen, y_screen