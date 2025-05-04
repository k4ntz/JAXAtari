import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import os

class SpriteHandler:
    def preview_sprite_and_edit(self, path):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.sprite = np.load(os.path.join(MODULE_DIR, path))
        current_color = [255, 0, 0, 255]  # red

        # Create figure and axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Leave space for sliders

        im = ax.imshow(self.sprite, interpolation='nearest')
        ax.set_title("Click to paint | Press 's' to save")

        # Axes for sliders
        ax_width = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_height = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_offset_x = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_offset_y = plt.axes([0.25, 0.2, 0.65, 0.03])

        slider_x_offset = Slider(ax_offset_x, 'X-Offset', -self.sprite.shape[1], self.sprite.shape[1], valinit=0, valstep=1)
        slider_y_offset = Slider(ax_offset_y, 'Y-Offset', -self.sprite.shape[0], self.sprite.shape[0], valinit=0, valstep=1)
        slider_width = Slider(ax_width, 'Width', 8, self.sprite.shape[1] * 2, valinit=self.sprite.shape[1], valstep=1)
        slider_height = Slider(ax_height, 'Height', 8, self.sprite.shape[0] * 2, valinit=self.sprite.shape[0], valstep=1)

        def resize_sprite_padcrop(new_height, new_width):
            h, w, c = self.sprite.shape
            padded = np.zeros((new_height, new_width, c), dtype=np.uint8)  # black background

            # Determine how much to copy
            copy_h = min(h, new_height)
            copy_w = min(w, new_width)

            padded[:copy_h, :copy_w] = self.sprite[:copy_h, :copy_w]
            return padded

        def apply_offset(x_offset, y_offset):
            h, w, c = self.sprite.shape
            shifted = np.zeros_like(self.sprite)

            # Compute ranges
            x_start_src = max(0, -x_offset)
            x_end_src = min(w, w - x_offset if x_offset >= 0 else w)
            x_start_dst = max(0, x_offset)
            x_end_dst = min(w, w + x_offset if x_offset < 0 else w)

            y_start_src = max(0, -y_offset)
            y_end_src = min(h, h - y_offset if y_offset >= 0 else h)
            y_start_dst = max(0, y_offset)
            y_end_dst = min(h, h + y_offset if y_offset < 0 else h)

            # Copy over valid region
            shifted[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = self.sprite[y_start_src:y_end_src, x_start_src:x_end_src]
            return shifted

        def update(val):
            new_off_x = int(slider_x_offset.val)
            new_off_y = int(slider_y_offset.val)
            new_w = int(slider_width.val)
            new_h = int(slider_height.val)
            self.sprite = resize_sprite_padcrop(new_h, new_w)
            sprite = apply_offset(new_off_x, new_off_y)
            im.set_data(sprite)
            im.set_extent((0, sprite.shape[1], sprite.shape[0], 0))
            fig.canvas.draw_idle()

        slider_x_offset.on_changed(update)
        slider_y_offset.on_changed(update)
        slider_width.on_changed(update)
        slider_height.on_changed(update)

        def onclick(event):
            if event.inaxes != ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= y < self.sprite.shape[0] and 0 <= x < self.sprite.shape[1]:
                self.sprite[y, x] = current_color
                im.set_data(self.sprite)
                plt.draw()

        def onkey(event):
            if event.key == 's':
                x_offset = int(slider_x_offset.val)
                y_offset = int(slider_y_offset.val)
                self.sprite = resize_sprite_padcrop(self.sprite, int(slider_height.val), int(slider_width.val))
                output = apply_offset(x_offset, y_offset)
                np.save("edited_sprite.npy", output)
                print("Sprite saved to 'edited_sprite.npy'")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)

        plt.show()