from tkinter import simpledialog
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.colorchooser import askcolor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk


class NPYImageEditor:
    def __init__(self, master):
        self.root = master
        self.root.title("NPY Image Editor")

        self.current_mouse_position = tk.StringVar()
        self.current_mouse_position.set("x=--, y=--")

        self.image = None
        self.zoom_level = 20
        self.current_color = [0, 0, 0, 255]  # Default color: black with full opacity
        self.selected_rgba_color = self.current_color  # Store the selected RGBA color

        self.mouse_pressed = False
        self.create_widgets()
        self.tool = None  # Initialize tool attribute
        self.selected = None
        self.selection_start = None
        self.selection_end = None
        self.state_queue = []
        self.current_state_index = -1
        self.default_canvas_size = (600, 400)  # Default canvas size (width, height)
        self.cmd_pressed = False  # For multi-select with magic wand

        # Bind keyboard shortcuts
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-a>", self.select_all)
        self.root.bind("<Control-d>", self.deselect_all)
        self.root.bind("<Control-s>", self.save_selection)
        # ctrl + scroll to zoom
        self.root.bind("<Control-MouseWheel>", self.on_mouse_scroll)
        # delete key to delete selected pixels
        self.root.bind("<Delete>", self.delete_selected)

        # circumvents a CTRL error I encountered
        self.root.bind("<Key>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

        # State Queue UI with Scrollbar
        self.state_queue_frame = tk.Frame(self.root)
        self.state_queue_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # Add a canvas for the state queue with an associated scrollbar
        self.state_canvas = tk.Canvas(self.state_queue_frame, width=200, height=300)
        self.state_canvas.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        self.scrollbar = tk.Scrollbar(
            self.state_queue_frame, orient=tk.VERTICAL, command=self.state_canvas.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.state_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.image_thumbnails = []  # List to store thumbnails for each state
        self.root = master
        self.root.title("NPY Image Editor")

        self.image = None
        self.zoom_level = 1.0
        self.mouse_pressed = False
        self.tool = None  # Initialize tool attribute
        self.selected = None
        self.selection_start = None
        self.selection_end = None
        self.selection_mode = "new"  # Initialize selection mode
        self.state_queue = []
        self.current_state_index = -1
        self.selection_preset = None

    def on_key_press(self, event):
        if event.keysym in ('Control_L', 'Control_R', 'Control'):
            self.cmd_pressed = True

    def on_key_release(self, event):
        if event.keysym in ('Control_L', 'Control_R', 'Control'):
            self.cmd_pressed = False

    def create_widgets(self):
        # Menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(
            label="Save Selection", command=lambda: self.save_selection(None)
        )
        file_menu.add_command(
            label="Save full image", command=lambda: self.save_full_image(None)
        )
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_menu)
        edit_menu.add_command(label="Redo", command=self.redo_menu)

        # Scrollable Canvas
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(self.canvas_frame)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")  # Use grid layout for proper alignment

        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")  # Place the horizontal scrollbar at the bottom

        self.v_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")  # Place the vertical scrollbar on the right

        self.canvas_frame.grid_rowconfigure(0, weight=1)  # Allow the canvas to expand vertically
        self.canvas_frame.grid_columnconfigure(0, weight=1)  # Allow the canvas to expand horizontally

        self.scroll_canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        # Embed Matplotlib Figure in Scrollable Canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self.scroll_canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_window = self.scroll_canvas.create_window(0, 0, anchor=tk.NW, window=self.canvas_widget)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)

        # Update scroll region when the canvas size changes
        self.scroll_canvas.bind("<Configure>", self.update_scroll_region)


        # Tool Buttons
        tools_frame = tk.Frame(self.root)
        tools_frame.pack(fill=tk.X)

        tk.Button(tools_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Zoom Out", command=self.zoom_out).pack(
            side=tk.LEFT
        )
        tk.Button(tools_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT)
    
        tk.Button(tools_frame, text="Pencil", command=self.activate_pencil).pack(
            side=tk.LEFT
        )
        tk.Button(
            tools_frame, text="Magic Wand", command=self.activate_magic_wand
        ).pack(side=tk.LEFT)
        tk.Button(
            tools_frame,
            text="Rectangular Selection",
            command=self.activate_rectangular_selection,
        ).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Dropper", command=self.activate_dropper).pack(
            side=tk.LEFT
        )
        tk.Button(
            tools_frame, text="Select w/ Same Color", command=self.select_all_with_color
        ).pack(side=tk.LEFT)

        # Current Color Indicator
        self.color_indicator = tk.Label(
            tools_frame,
            text="",
            bg=self.rgb_to_hex(self.current_color[:3]),
            width=5,
            relief=tk.RAISED,
        )
        self.color_indicator.pack(side=tk.LEFT, padx=5)
        self.color_indicator.bind("<Button-1>", self.open_color_palette)

        # Current Alpha Indicator
        self.alpha_indicator = tk.Label(
            tools_frame, text=f"a: {self.current_color[3]}", width=5, relief=tk.RAISED
        )
        self.alpha_indicator.pack(side=tk.LEFT, padx=5)
        self.alpha_indicator.bind("<Button-1>", self.open_alpha_input)

        self.tool = None

        # Second row for Selection Mode (hidden by default)
        self.selection_mode_frame = tk.Frame(self.root)
        self.selection_mode_frame.pack(fill=tk.X)
        self.selection_mode_frame.pack_forget()  # Initially hide this frame

        tk.Label(self.selection_mode_frame, text="Selection Mode:").pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(
            self.selection_mode_frame,
            text="New",
            command=lambda: self.set_selection_mode("new"),
        ).pack(side=tk.LEFT)
        tk.Button(
            self.selection_mode_frame,
            text="Add",
            command=lambda: self.set_selection_mode("add"),
        ).pack(side=tk.LEFT)
        tk.Button(
            self.selection_mode_frame,
            text="Subtract",
            command=lambda: self.set_selection_mode("subtract"),
        ).pack(side=tk.LEFT)
        tk.Button(
            self.selection_mode_frame,
            text="Intersect",
            command=lambda: self.set_selection_mode("intersect"),
        ).pack(side=tk.LEFT)
        tk.Button(
            self.selection_mode_frame,
            text="Fill Selected",
            command=lambda: self.fill_selected(),
        ).pack(side=tk.LEFT)
        tk.Button(
            self.selection_mode_frame,
            text="Save as Preset",
            command=lambda: self.save_selection_preset(),
        ).pack(side=tk.LEFT)
        tk.Button(
            self.selection_mode_frame,
            text="Load from Preset",
            command=lambda: self.load_selection_preset(),
        ).pack(side=tk.LEFT)

        status_bar = tk.Label(self.root, textvariable=self.current_mouse_position, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_selection_mode(self, mode):
        self.selection_mode = mode

    def on_cmd_press(self, event):
        print(f"Control pressed: {event.keysym}")
        self.cmd_pressed = True

    def on_cmd_release(self, event):
        print(f"Control released: {event.keysym}")
        self.cmd_pressed = False

    def fill_selected(self):
        if self.selected is not None and self.image is not None:
            self.image[self.selected] = self.current_color
            self.update_display()
            self.update_state("fill_selected")

    def update_state(self, last_step_name=None):

        new_state = self.get_current_state(last_step_name)

        if self.current_state_index < len(self.state_queue) - 1:
            self.state_queue = self.state_queue[0 : self.current_state_index + 1]

        self.state_queue.append(new_state)

        self.current_state_index = len(self.state_queue) - 1

        self.update_display()
        self.update_state_queue_display()

    def update_state_queue_display(self):
        self.state_canvas.delete("all")  # Clear the canvas before drawing new items

        # Clear the image_thumbnails list before creating new ones
        self.image_thumbnails = []

        y_position = 10  # Start from top position for the first state

        for idx, state in enumerate(self.state_queue):
            image_thumb = self.create_thumbnail(state["image"])
            self.image_thumbnails.append(image_thumb)  # Store the thumbnail

            # Create an image for the thumbnail on the left
            self.state_canvas.create_image(
                10, y_position, anchor=tk.NW, image=image_thumb
            )

            # Set the text color (grey for states after the current state)
            text_color = "grey" if idx > self.current_state_index else "black"

            # Add text to the right of the image
            self.state_canvas.create_text(
                60,
                y_position + 10,
                anchor=tk.NW,
                text=state["last_step_name"],
                fill=text_color,
            )

            y_position += 60  # Move the next state a bit lower on the canvas

        # Automatically scroll to the bottom after updating
        self.state_canvas.config(scrollregion=self.state_canvas.bbox("all"))
        self.state_canvas.yview_moveto(1)  # Scroll to the bottom

        # Get the current state of the editor

    def get_current_state(self, last_step_name=None):
        return {
            # clone the reference type
            "image": self.image.copy(),
            "zoom_level": self.zoom_level,
            "current_color": self.current_color.copy(),
            "selected": self.selected.copy(),
            "last_step_name": self.tool if last_step_name is None else last_step_name,
        }

    # load the state to the editor
    def load_state(self, state):
        self.image = state["image"].copy()
        self.zoom_level = state["zoom_level"]
        self.current_color = state["current_color"].copy()
        self.selected = state["selected"].copy()
        self.update_display()

    # undo the last step
    def undo(self, _):
        if self.current_state_index > 0:
            # Go back one step
            self.current_state_index -= 1
            state_to_load = self.state_queue[self.current_state_index]

            # Load the previous state and update the state queue
            self.load_state(state_to_load)
            self.update_state_queue_display()
        else:
            print("No more steps to undo.")

    def undo_menu(self):
        self.undo(None)

    def redo(self, _):
        if self.current_state_index < len(self.state_queue) - 1:
            # Move forward one step
            self.current_state_index += 1
            state_to_load = self.state_queue[self.current_state_index]

            # Load the next state and update the state queue
            self.load_state(state_to_load)
            self.update_state_queue_display()
        else:
            print("No more steps to redo.")

    def redo_menu(self):
        self.redo(None)

        # Create a thumbnail of the image

    def create_thumbnail(self, image):
        # Resize the image to create a thumbnail (e.g., 50x50 pixels)
        img_pil = Image.fromarray(image)  # Convert NumPy array to a PIL Image
        img_resized = img_pil.resize((50, 50))  # Resize using interpolation

        # Create a Tkinter PhotoImage from the resized PIL image
        return ImageTk.PhotoImage(img_resized)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if not file_path:
            return

        try:
            # set window title as the file name
            self.root.title(f"NPY Image Editor - {file_path}")
            self.image = np.load(file_path)
            # Convert to RGBA if not already in RGBA format
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                # Convert to RGBA by adding an alpha channel (255 means fully opaque)
                print("RGB format detected. Converts to RGBA format.")
                self.image = np.dstack(
                    [self.image, np.ones(self.image.shape[:2], dtype=np.uint8) * 255]
                )

            elif len(self.image.shape) != 3 or self.image.shape[2] != 4:
                raise ValueError(
                    "Invalid NPY file format. Expected 3D array with 4 channels (RGBA)."
                )

            self.zoom_level = 1
            self.update_display()
            self.selected = np.zeros(self.image.shape[:2], dtype=bool)
            self.update_state("open file")
        except (ValueError, IOError) as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_selection(self, _):
        # get the minimal rectangle that contains all selected pixels
        y, x = np.where(self.selected)
        # if no pixel is selected, return
        if len(y) == 0:
            messagebox.showwarning(
                "Warning", "No pixel is selected. Please select a region to be saved."
            )
            return
        y0, y1 = np.min(y), np.max(y)
        x0, x1 = np.min(x), np.max(x)

        region_to_save = self.image[y0 : y1 + 1, x0 : x1 + 1].copy()
        selected_pixels = self.selected[y0 : y1 + 1, x0 : x1 + 1]
        # set the unselected pixels as transparent (a=0)
        region_to_save[~selected_pixels] = [0, 0, 0, 0]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy", filetypes=[("NumPy files", "*.npy")]
        )
        if file_path:
            np.save(file_path, region_to_save)
            messagebox.showinfo("Saved", f"Selection saved to {file_path}")
        else:
            messagebox.showwarning("Warning", "Invalid Path.")


    def save_full_image(self, _):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy", filetypes=[("NumPy files", "*.npy")]
        )
        if file_path:
            np.save(file_path, self.image)
            messagebox.showinfo("Saved", f"Image saved to {file_path}")
        else:
            messagebox.showwarning("Warning", "Invalid Path.")


    def select_all(self, _):
        if self.image is not None:
            self.selected = np.ones(
                self.image.shape[:2], dtype=bool
            )  # Select all pixels
            self.update_state("select all")

    def deselect_all(self, _):
        if self.image is not None:
            self.selected = np.zeros(
                self.image.shape[:2], dtype=bool
            )  # Deselect all pixels
            self.update_state("deselect")

    def zoom_in(self):
        self.zoom_level *= 1.5
        self.update_display()

    def zoom_out(self):
        self.zoom_level /= 1.5
        self.update_display()

    def reset_zoom(self):
        self.zoom_level = 1.0  # Reset zoom level to default
        self.update_display()  # Update the display to reflect the reset

    def activate_pencil(self):
        self.tool = "pencil"
        self.selection_mode_frame.pack_forget()  # Hide selection mode buttons

    def activate_magic_wand(self):
        self.tool = "magic_wand"
        self.selection_mode_frame.pack(fill=tk.X)  # Show selection mode buttons

    def activate_rectangular_selection(self):
        self.tool = "rectangular_selection"
        self.selection_mode_frame.pack(fill=tk.X)  # Show selection mode buttons

    def activate_dropper(self):
        self.tool = "dropper"
        self.selection_mode_frame.pack_forget()  # Hide selection mode buttons

    def select_all_with_color(self):
        self.tool = "select_all_with_color"
        self.selection_mode_frame.pack(fill=tk.X)  # Show selection mode buttons

    def save_selection_preset(self):
        if self.selected is not None:
            self.selection_preset = (
                self.selected.copy()
            )  # Save the current selection as a preset

    def load_selection_preset(self):
        if self.selection_preset is not None:
            self.selected = self.selection_preset.copy()

    def open_color_palette(self, event=None):
        # Open color chooser dialog
        color = askcolor(color=self.rgb_to_hex(self.current_color[:3]))[0]
        if color:
            # Convert the selected color to RGBA format
            r, g, b = color
            a = self.current_color[3]  # keep the alpha value
            self.current_color = [
                int(r),
                int(g),
                int(b),
                a,
            ]  # Update color in RGBA format
            self.color_indicator.config(
                bg=self.rgb_to_hex(self.current_color)
            )  # Update color indicator

    def open_alpha_input(self, event=None):
        # open a dialog to input the alpha value (0~255)
        alpha = simpledialog.askinteger(
            "Input",
            "Enter an integer between 0 and 255",
            parent=self.root,
            minvalue=0,
            maxvalue=255,
        )
        if alpha is not None:
            self.current_color[3] = alpha
            self.alpha_indicator.config(text=f"a: {alpha}")
            self.color_indicator.config(bg=self.rgb_to_hex(self.current_color[:3]))

    def on_mouse_press(self, event):
        if self.image is None or event.xdata is None or event.ydata is None:
            return

        self.mouse_pressed = True
        x, y = int(event.xdata), int(event.ydata)

        if self.tool == "pencil":
            self.image[y, x] = self.current_color
            self.update_display()
        elif self.tool == "rectangular_selection":
            # mark the starting point of the selection
            self.selection_start = (y, x)
        elif self.tool == "dropper":
            self.current_color = self.image[y, x].tolist()
            self.color_indicator.config(bg=self.rgba_to_hex(self.current_color))
            self.alpha_indicator.config(text=f"a: {self.current_color[3]}")

    def on_mouse_release(self, event):  # Added to keep track of mouse button state

        if self.mouse_pressed and event.xdata and event.ydata:
            if self.tool == "rectangular_selection":
                self.selection_end = (int(event.ydata), int(event.xdata))
                # mark the rectangle between selection_start and selection_end as selected
                y0, x0 = self.selection_start
                y1, x1 = self.selection_end
                selected_pixels = np.zeros(self.image.shape[:2], dtype=bool)
                selected_pixels[y0 : y1 + 1, x0 : x1 + 1] = True
                self.submit_selection(selected_pixels)
                self.update_state("rectangular_selection")

            if self.tool == "pencil":
                self.update_state("pencil")
            if self.tool == "magic_wand":
                new_selection = self.magic_wand(
                    int(event.ydata),
                    int(event.xdata),
                    self.image[int(event.ydata), int(event.xdata)],
                )

                # If CMD/CTRL is pressed, add to current selection instead of replacing it
                if self.cmd_pressed:
                    # Add new selection to current selection by logical or on the whole array
                    for i in range(len(self.selected)):
                        for j in range(len(self.selected[0])):
                            if new_selection[i, j]:
                                self.selected[i, j] = True

                else:
                    # Normal behavior - replace selection
                    self.selected = new_selection

                self.update_state("magic_wand")

            if self.tool == "select_all_with_color":
                new_selection = np.all(
                    self.image == self.image[int(event.ydata), int(event.xdata)], axis=2
                )
                self.submit_selection(new_selection)
                self.update_state("select with same color")

        self.selection_start = None
        self.selection_end = None
        self.mouse_pressed = False

    # delete the selected region. Set the alpha as 0
    def delete_selected(self, _):
        if self.selected is not None:
            self.image[self.selected] = [0, 0, 0, 0]
            self.update_display()
            self.update_state("delete selected")

    def on_mouse_scroll(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # DFS to find all connected pixels with the same color
    def magic_wand(self, y, x, target_color):
        stack = [(y, x)]
        new_selection = np.zeros(self.image.shape[:2], dtype=bool)

        # Ensure the initial pixel is valid
        if not (0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]):
            return new_selection

        # Mark the clicked pixel as selected
        new_selection[y, x] = True

        # Then find all connected pixels of the same color
        processed = {(y, x)}  # Track processed pixels to avoid duplicates

        while stack:
            cur_y, cur_x = stack.pop()

            # Check the four adjacent pixels
            for ny, nx in [(cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)]:
                if (not (0 <= ny < self.image.shape[0] and 0 <= nx < self.image.shape[1])
                        or (ny, nx) in processed):
                    continue

                processed.add((ny, nx))

                # If the color matches, mark as selected and continue DFS
                if np.all(self.image[ny, nx] == target_color):
                    new_selection[ny, nx] = True
                    stack.append((ny, nx))

        return new_selection

    def on_mouse_motion(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.current_mouse_position.set(f"x={x}, y={y}")
            if self.tool == "pencil" and self.mouse_pressed:
                self.image[y, x] = self.current_color

            if self.tool == "rectangular_selection" and self.mouse_pressed:
                self.selection_end = (int(event.ydata), int(event.xdata))
        else:
            self.current_mouse_position.set("x=--, y=--")

        self.update_display()

    def update_canvas_size(self):
        if self.image is not None:
            original_height, original_width = self.image.shape[:2]

            # Calculate new dimensions based on zoom level
            new_width = int(original_width * self.zoom_level)
            new_height = int(original_height * self.zoom_level)

            # Update canvas widget size
            self.canvas_widget.config(width=new_width, height=new_height)

            # Update scroll region
            self.scroll_canvas.itemconfig(self.canvas_window, width=new_width, height=new_height)
            self.scroll_canvas.config(scrollregion=(0, 0, new_width, new_height))

    def update_scroll_region(self, event=None):
        self.scroll_canvas.config(scrollregion=self.scroll_canvas.bbox("all"))

    def update_display(self):
        if self.image is None:
            return

        self.ax.clear()
        self.ax.axis("off")  # Hide axes for a cleaner view

        # Display the base image

        # Resize the canvas to match the new image size
        self.update_canvas_size()

        self.ax.imshow(self.image)

        # Mark the current rectangular selection with a rectangle border
        if self.selection_start and self.selection_end:
            y0, x0 = self.selection_start
            y1, x1 = self.selection_end
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="r", facecolor="none"
            )
            self.ax.add_patch(rect)

        # Highlight the selected region without darkening other pixels
        if self.selected is not None and self.selected.any():
            # Create an RGBA overlay image with transparency
            overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
            overlay[self.selected] = [0, 0, 255, 128]  # Blue with 50% opacity

            # Display the overlay on top of the image
            self.ax.imshow(overlay, interpolation="none")

        self.canvas.draw()

    def rgba_to_hex(self, rgba):
        # Convert rgba to floats
        rgba = np.array(rgba) / 255
        # Adjust RGB with respect to alpha value
        rgb = (1 - rgba[3]) + rgba[3] * rgba[:3]
        # Convert RGB back to 0-255 range
        rgb = (rgb * 255).astype(int)
        # Convert RGB to Hex format
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def rgb_to_hex(self, rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def submit_selection(self, selection):
        if self.selection_mode == "new":
            self.selected = selection
        elif self.selection_mode == "add":
            self.selected = np.logical_or(self.selected, selection)
        elif self.selection_mode == "subtract":
            self.selected = np.logical_and(self.selected, np.logical_not(selection))
        elif self.selection_mode == "intersect":
            self.selected = np.logical_and(self.selected, selection)


if __name__ == "__main__":
    root = tk.Tk()
    app = NPYImageEditor(root)
    root.mainloop()
