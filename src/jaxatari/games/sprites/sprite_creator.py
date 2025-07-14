import tkinter as tk
from tkinter import filedialog, simpledialog, colorchooser
import numpy as np
from PIL import Image


class SpriteEditor:
    def __init__(self, master, grid_width, grid_height, pixel_size):
        self.master = master
        self.master.title("Sprite Editor mit Transparenz")

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.pixel_size = pixel_size

        self.canvas = tk.Canvas(master, width=self.grid_width * self.pixel_size,
                                height=self.grid_height * self.pixel_size, bg="white")
        self.canvas.pack()

        self.sprite = np.zeros((self.grid_height, self.grid_width, 4), dtype=np.uint8)
        self.sprite[:, :, 3] = 255  # Alpha: vollständig sichtbar

        self.current_color = [0, 0, 0, 255]
        self.eraser_mode = False

        self.canvas.bind("<Button-1>", self.paint_pixel)

        control_frame = tk.Frame(master)
        control_frame.pack()

        # Farbwahl-Button
        tk.Button(control_frame, text="Farbe wählen", command=self.choose_color).pack(side=tk.LEFT)

        # Alpha-Slider
        self.alpha_slider = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Transparenz (Alpha)")
        self.alpha_slider.set(255)
        self.alpha_slider.pack(side=tk.LEFT)

        # Radiergummi
        tk.Button(control_frame, text="Radiergummi", command=self.toggle_eraser).pack(side=tk.LEFT)

        # Speichern
        tk.Button(control_frame, text="Speichern als .npy", command=self.save_sprite).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Export als PNG", command=self.save_png).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Export als Python-Code", command=self.export_as_python_code).pack(side=tk.LEFT)

    def choose_color(self):
        rgb_color, _ = colorchooser.askcolor()
        if rgb_color:
            r, g, b = map(int, rgb_color)
            a = self.alpha_slider.get()
            self.current_color = [r, g, b, a]
            self.eraser_mode = False

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.current_color = [0, 0, 0, 0]
        else:
            # Stelle aktuelle Farbe mit aktueller Transparenz wieder her
            r, g, b = self.current_color[:3]
            a = self.alpha_slider.get()
            self.current_color = [r, g, b, a]

    def paint_pixel(self, event):
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            if not self.eraser_mode:
                self.current_color[3] = self.alpha_slider.get()
            self.sprite[y, x] = self.current_color
            self.draw_pixel(x, y, self.current_color)

    def draw_pixel(self, x, y, color):
        r, g, b, a = color
        if a == 0:
            fill = "#ffffff"
        else:
            fill = f"#{r:02x}{g:02x}{b:02x}"

        self.canvas.create_rectangle(
            x * self.pixel_size, y * self.pixel_size,
            (x + 1) * self.pixel_size, (y + 1) * self.pixel_size,
            fill=fill,
            outline="gray"  # <- Hier kannst du outline="" setzen, um Rasterlinien zu entfernen
        )

    def save_sprite(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy")
        if file_path:
            np.save(file_path, self.sprite)
            print("Sprite gespeichert:", file_path)

    def save_png(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            img = Image.fromarray(self.sprite, mode='RGBA')
            img.save(file_path)
            print("Bild gespeichert:", file_path)

    def export_as_python_code(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".py")
        if file_path:
            with open(file_path, "w") as f:
                f.write("sprite = np.zeros(({}, {}, 4), dtype=np.uint8)\n\n".format(
                    self.grid_height, self.grid_width))
                for y in range(self.grid_height):
                    for x in range(self.grid_width):
                        r, g, b, a = self.sprite[y, x]
                        if a > 0 and (r != 0 or g != 0 or b != 0):  # Nur wirklich sichtbare Pixel
                            f.write(f"sprite[{y}, {x}] = [{r}, {g}, {b}, {a}]\n")

            print("Python-Code exportiert:", file_path)


def main():
    root = tk.Tk()

    # Projektgröße beim Start abfragen
    width = simpledialog.askinteger("Breite", "Anzahl Pixel in der Breite:", initialvalue=8, minvalue=1)
    height = simpledialog.askinteger("Höhe", "Anzahl Pixel in der Höhe:", initialvalue=8, minvalue=1)
    pixel_size = simpledialog.askinteger("Pixelgröße", "Größe jedes Pixels (z. B. 20):", initialvalue=30, minvalue=1)

    if width and height and pixel_size:
        app = SpriteEditor(root, width, height, pixel_size)
        root.mainloop()


if __name__ == "__main__":
    main()
