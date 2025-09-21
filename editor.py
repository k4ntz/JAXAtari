import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
from PIL import Image, ImageTk
import os
from collections import deque
from scipy.ndimage import label


class PixelEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Editor - .npy Files")
        self.root.geometry("1200x800")
        
        self.image_array = None
        self.display_image = None
        self.photo_image = None
        self.original_size = None
        self.current_file_name = None
        
        self.zoom_level = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        self.brush_color = (255, 0, 0)
        self.show_grid = True
        self.is_painting = False
        self.pipette_mode = False
        self.current_tool = "brush"
        
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        self.selection_mask = None
        self.selection_overlay = None
        
        self.rect_start_x = None
        self.rect_start_y = None
        self.is_selecting = False
        self.selection_rect = None
        
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50
        
        # Performance caching
        self.cached_selection_rects = []
        self.last_zoom_level = None
        
        self.setup_ui()
        
    def setup_ui(self):
        self.create_menu_bar()
        
        # File name display at top
        file_frame = ttk.Frame(self.root)
        file_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.file_name_label = ttk.Label(file_frame, text="No file loaded", anchor="w", font=('Arial', 10, 'bold'))
        self.file_name_label.pack(fill=tk.X)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_sidebar(main_frame)
        self.create_canvas_area(main_frame)
        
    def create_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load .npy", command=self.load_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save as .npy", command=self.save_npy, accelerator="Ctrl+S")
        file_menu.add_command(label="Save as .png", command=self.save_png, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Save Selection", command=self.save_selection, accelerator="Ctrl+Alt+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Selection", command=self.clear_selection, accelerator="Ctrl+D")
        
        self.root.bind('<Control-o>', lambda e: self.load_file())
        self.root.bind('<Control-s>', lambda e: self.save_npy())
        self.root.bind('<Control-Shift-S>', lambda e: self.save_png())
        self.root.bind('<Control-Alt-s>', lambda e: self.save_selection())
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-d>', lambda e: self.clear_selection())
        
    def create_sidebar(self, parent):
        # Create scrollable sidebar
        sidebar_container = ttk.Frame(parent)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Canvas for scrolling
        sidebar_canvas = tk.Canvas(sidebar_container, width=200, highlightthickness=0)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.Y)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame inside canvas
        sidebar = ttk.Frame(sidebar_canvas)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        
        def configure_sidebar_scroll(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
            # Update canvas window width to match canvas width
            canvas_width = sidebar_canvas.winfo_width()
            sidebar_canvas.itemconfig(sidebar_window, width=canvas_width)
        
        def configure_canvas_width(event):
            canvas_width = sidebar_canvas.winfo_width()
            sidebar_canvas.itemconfig(sidebar_window, width=canvas_width)
        
        sidebar.bind('<Configure>', configure_sidebar_scroll)
        sidebar_canvas.bind('<Configure>', configure_canvas_width)
        
        # Bind mouse wheel to canvas for scrolling
        def on_mousewheel(event):
            sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            sidebar_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            sidebar_canvas.unbind_all("<MouseWheel>")
        
        sidebar_canvas.bind('<Enter>', bind_mousewheel)
        sidebar_canvas.bind('<Leave>', unbind_mousewheel)
        
        # Now add all the sidebar content
        ttk.Label(sidebar, text="Tools", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        tools_frame = ttk.LabelFrame(sidebar, text="Tool Selection", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.tool_var = tk.StringVar(value="brush")
        ttk.Radiobutton(tools_frame, text="Brush", variable=self.tool_var, 
                       value="brush", command=self.change_tool).pack(anchor='w')
        ttk.Radiobutton(tools_frame, text="Color Pipette", variable=self.tool_var, 
                       value="pipette", command=self.change_tool).pack(anchor='w')
        ttk.Radiobutton(tools_frame, text="Single Pixel Select", variable=self.tool_var, 
                       value="single_select", command=self.change_tool).pack(anchor='w')
        ttk.Radiobutton(tools_frame, text="Rectangular Select", variable=self.tool_var, 
                       value="rect_select", command=self.change_tool).pack(anchor='w')
        ttk.Radiobutton(tools_frame, text="Magic Wand", variable=self.tool_var, 
                       value="magic_wand", command=self.change_tool).pack(anchor='w')
        ttk.Radiobutton(tools_frame, text="Object Select", variable=self.tool_var, 
                       value="object_select", command=self.change_tool).pack(anchor='w')
        
        color_frame = ttk.LabelFrame(sidebar, text="Brush Color", padding=10)
        color_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.color_display = tk.Canvas(color_frame, width=50, height=30, bg='red')
        self.color_display.pack(pady=(0, 5))
        
        ttk.Button(color_frame, text="Pick Color", command=self.pick_color).pack(fill=tk.X)
        
        self.rgb_label = ttk.Label(color_frame, text="RGB: (255, 0, 0)")
        self.rgb_label.pack(pady=(5, 0))
        
        selection_frame = ttk.LabelFrame(sidebar, text="Selection", padding=10)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(selection_frame, text="Clear Selection", command=self.clear_selection).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(selection_frame, text="Save Selection", command=self.save_selection).pack(fill=tk.X)
        
        self.selection_info = ttk.Label(selection_frame, text="No selection")
        self.selection_info.pack(pady=(5, 0))
        
        zoom_frame = ttk.LabelFrame(sidebar, text="Zoom", padding=10)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        zoom_buttons_frame = ttk.Frame(zoom_frame)
        zoom_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(zoom_buttons_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_buttons_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.RIGHT)
        
        # Reset button in new row
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(fill=tk.X, pady=(5, 0))
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(pady=(5, 0))
        
        ttk.Label(zoom_frame, text="Ctrl+Wheel to zoom", font=('Arial', 8)).pack()
        ttk.Label(zoom_frame, text="Right-click to pan", font=('Arial', 8)).pack()
        
        grid_frame = ttk.LabelFrame(sidebar, text="Display", padding=10)
        grid_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid_frame, text="Show Pixel Grid", 
                       variable=self.grid_var, command=self.toggle_grid).pack()
        
        # Mouse position display
        self.mouse_pos_label = ttk.Label(grid_frame, text="Mouse: -", font=('Arial', 8))
        self.mouse_pos_label.pack(pady=(5, 0))
        
        info_frame = ttk.LabelFrame(sidebar, text="Image Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="No image loaded", wraplength=180)
        self.info_label.pack()
        
    def create_canvas_area(self, parent):
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='lightgray')
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        
        self.canvas.bind("<Button-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.pan_canvas)
        self.canvas.bind("<ButtonRelease-3>", self.stop_pan)
        
        self.canvas.bind("<Control-MouseWheel>", self.mouse_wheel_zoom)
        self.canvas.bind("<Control-Button-4>", self.mouse_wheel_zoom)
        self.canvas.bind("<Control-Button-5>", self.mouse_wheel_zoom)
        
        # Mouse motion tracking
        self.canvas.bind("<Motion>", self.canvas_motion)
        
        self.canvas.focus_set()
        
    def canvas_motion(self, event):
        """Track mouse position and update display"""
        if self.image_array is not None:
            pixel_x, pixel_y = self.canvas_to_pixel(event.x, event.y)
            if pixel_x is not None and pixel_y is not None:
                self.mouse_pos_label.configure(text=f"Mouse: ({pixel_x}, {pixel_y})")
            else:
                self.mouse_pos_label.configure(text="Mouse: -")
        else:
            self.mouse_pos_label.configure(text="Mouse: -")
        
    def change_tool(self):
        self.current_tool = self.tool_var.get()
        
        cursors = {
            "brush": "",
            "pipette": "dotbox",
            "single_select": "crosshair",
            "rect_select": "crosshair",
            "magic_wand": "target",
            "object_select": "hand2"
        }
        self.canvas.configure(cursor=cursors.get(self.current_tool, ""))
        
    def start_pan(self, event):
        if self.image_array is not None:
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.canvas.configure(cursor="fleur")
            self.canvas.scan_mark(event.x, event.y)
    
    def pan_canvas(self, event):
        if self.is_panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def stop_pan(self, event):
        self.is_panning = False
        self.change_tool()
        
    def mouse_wheel_zoom(self, event):
        if self.image_array is None:
            return
            
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if event.delta > 0 or event.num == 4:
            zoom_factor = 1.2
        else:
            zoom_factor = 1.0 / 1.2
        
        old_zoom = self.zoom_level
        new_zoom = old_zoom * zoom_factor
        new_zoom = max(0.1, min(50.0, new_zoom))
        
        if new_zoom != old_zoom:
            pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
            
            self.zoom_level = new_zoom
            self.update_display()
            self.update_zoom_display()
            
            if pixel_x is not None and pixel_y is not None:
                new_canvas_x = self.canvas_offset_x + pixel_x * self.zoom_level
                new_canvas_y = self.canvas_offset_y + pixel_y * self.zoom_level
                
                dx = new_canvas_x - event.x
                dy = new_canvas_y - event.y
                
                bbox = self.canvas.bbox("all")
                if bbox:
                    scroll_width = bbox[2] - bbox[0]
                    scroll_height = bbox[3] - bbox[1]
                    
                    if scroll_width > 0:
                        current_x = self.canvas.canvasx(0)
                        new_scroll_x = (current_x + dx) / scroll_width
                        self.canvas.xview_moveto(max(0, min(1, new_scroll_x)))
                    
                    if scroll_height > 0:
                        current_y = self.canvas.canvasy(0)
                        new_scroll_y = (current_y + dy) / scroll_height
                        self.canvas.yview_moveto(max(0, min(1, new_scroll_y)))
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Load .npy file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.image_array = np.load(file_path)
            
            if self.image_array.ndim not in [2, 3]:
                messagebox.showerror("Error", "Image must be 2D (grayscale) or 3D (RGB/RGBA)")
                return
                
            if self.image_array.ndim == 2:
                self.image_array = np.stack([self.image_array] * 3, axis=-1)
            
            if self.image_array.dtype != np.uint8:
                if self.image_array.max() <= 1.0:
                    self.image_array = (self.image_array * 255).astype(np.uint8)
                else:
                    self.image_array = self.image_array.astype(np.uint8)
            
            if self.image_array.shape[2] not in [3, 4]:
                messagebox.showerror("Error", "Image must have 3 (RGB) or 4 (RGBA) channels")
                return
                
            self.original_size = self.image_array.shape[:2]
            
            self.selection_mask = np.zeros(self.original_size, dtype=bool)
            
            self.undo_stack = []
            self.redo_stack = []
            
            initial_state = {
                'image': self.image_array.copy(),
                'selection': self.selection_mask.copy()
            }
            self.undo_stack.append(initial_state)
            self.save_state()
            
            # Update file name display
            self.current_file_name = os.path.basename(file_path)
            self.file_name_label.configure(text=f"File: {self.current_file_name}")
            
            self.fit_image_to_canvas()
            self.update_image_info()
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            
    def fit_image_to_canvas(self):
        if self.image_array is None:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.fit_image_to_canvas)
            return
            
        img_height, img_width = self.original_size
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height
        
        self.zoom_level = min(zoom_x, zoom_y) * 0.9
        self.zoom_level = max(self.zoom_level, 0.1)
        
        self.update_zoom_display()

    def reset_zoom(self):
        if self.image_array is not None:
            self.fit_image_to_canvas()
            self.update_display()
        
    def save_npy(self):
        if self.image_array is None:
            messagebox.showwarning("Warning", "No image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save as .npy",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                np.save(file_path, self.image_array)
                # Update file name display if saving with new name
                self.current_file_name = os.path.basename(file_path)
                self.file_name_label.configure(text=f"File: {self.current_file_name}")
                messagebox.showinfo("Success", "Image saved as .npy file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
                
    def save_png(self):
        if self.image_array is None:
            messagebox.showwarning("Warning", "No image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save as .png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if self.image_array.shape[2] == 3:
                    mode = 'RGB'
                else:
                    mode = 'RGBA'
                    
                img = Image.fromarray(self.image_array, mode)
                img.save(file_path)
                messagebox.showinfo("Success", "Image saved as .png file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
                
    def save_selection(self):
        if self.image_array is None or self.selection_mask is None:
            messagebox.showwarning("Warning", "No image or selection to save")
            return
            
        if not np.any(self.selection_mask):
            messagebox.showwarning("Warning", "No pixels selected")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save selection as .npy",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                selected_rows, selected_cols = np.where(self.selection_mask)
                min_row, max_row = selected_rows.min(), selected_rows.max()
                min_col, max_col = selected_cols.min(), selected_cols.max()
                
                selection_box = self.image_array[min_row:max_row+1, min_col:max_col+1]
                mask_box = self.selection_mask[min_row:max_row+1, min_col:max_col+1]
                
                if self.image_array.shape[2] == 3:
                    output = np.zeros((mask_box.shape[0], mask_box.shape[1], 4), dtype=np.uint8)
                    output[mask_box, :3] = selection_box[mask_box]
                    output[mask_box, 3] = 255
                else:
                    output = selection_box.copy()
                    output[~mask_box, 3] = 0
                
                np.save(file_path, output)
                
                box_width = max_col - min_col + 1
                box_height = max_row - min_row + 1
                num_selected = np.sum(self.selection_mask)
                
                messagebox.showinfo("Success", 
                    f"Selection saved as .npy file\n"
                    f"Bounding box: {box_width}x{box_height}\n"
                    f"Selected pixels: {num_selected}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save selection: {str(e)}")
                
    def pick_color(self):
        color = colorchooser.askcolor(title="Choose brush color")
        if color[0]:
            self.brush_color = tuple(int(c) for c in color[0])
            
            hex_color = f"#{int(color[0][0]):02x}{int(color[0][1]):02x}{int(color[0][2]):02x}"
            self.color_display.configure(bg=hex_color)
            self.rgb_label.configure(text=f"RGB: {self.brush_color}")
            
    def zoom_in(self):
        if self.image_array is not None:
            self.zoom_level = min(self.zoom_level * 1.5, 50.0)
            self.update_display()
            self.update_zoom_display()
            
    def zoom_out(self):
        if self.image_array is not None:
            self.zoom_level = max(self.zoom_level / 1.5, 0.1)
            self.update_display()
            self.update_zoom_display()
            
    def update_zoom_display(self):
        zoom_percent = int(self.zoom_level * 100)
        self.zoom_label.configure(text=f"{zoom_percent}%")
        
    def toggle_grid(self):
        self.show_grid = self.grid_var.get()
        self.update_display()
        
    def update_display(self):
        if self.image_array is None:
            return
            
        try:
            if self.image_array.shape[2] == 3:
                pil_image = Image.fromarray(self.image_array, 'RGB')
            else:
                pil_image = Image.fromarray(self.image_array, 'RGBA')
                
            new_width = int(self.original_size[1] * self.zoom_level)
            new_height = int(self.original_size[0] * self.zoom_level)
            
            scaled_image = pil_image.resize((new_width, new_height), Image.NEAREST)
            
            self.photo_image = ImageTk.PhotoImage(scaled_image)
            
            self.canvas.delete("all")
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            x = max(0, (canvas_width - new_width) // 2)
            y = max(0, (canvas_height - new_height) // 2)
            
            self.canvas_offset_x = x
            self.canvas_offset_y = y
            
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image, tags="image")
            
            if self.selection_mask is not None:
                self.draw_selection_overlay_optimized(x, y, new_width, new_height)
            
            if self.show_grid and self.zoom_level >= 2:
                self.draw_grid(x, y, new_width, new_height)
                
            scroll_region = (0, 0, max(canvas_width, new_width + x), max(canvas_height, new_height + y))
            self.canvas.configure(scrollregion=scroll_region)
            
        except Exception as e:
            print(f"Display update error: {e}")
            
    def draw_grid(self, start_x, start_y, width, height):
        pixel_size = self.zoom_level
        
        for i in range(int(width / pixel_size) + 1):
            x = start_x + i * pixel_size
            self.canvas.create_line(x, start_y, x, start_y + height, 
                                  fill="gray40", width=1, tags="grid", stipple="gray50")
                                  
        for i in range(int(height / pixel_size) + 1):
            y = start_y + i * pixel_size
            self.canvas.create_line(start_x, y, start_x + width, y, 
                                  fill="gray40", width=1, tags="grid", stipple="gray50")
    
    def draw_selection_overlay_optimized(self, start_x, start_y, width, height):
        if not np.any(self.selection_mask):
            return
            
        pixel_size = self.zoom_level
        
        selected_coords = np.where(self.selection_mask)
        
        for i in range(len(selected_coords[0])):
            row = selected_coords[0][i]
            col = selected_coords[1][i]
            
            x1 = start_x + col * pixel_size
            y1 = start_y + row * pixel_size
            x2 = x1 + pixel_size
            y2 = y1 + pixel_size
            
            self.canvas.create_rectangle(x1, y1, x2, y2, 
                                       fill="blue", stipple="gray50", 
                                       outline="blue", width=1, tags="selection")
                                               
    def clear_selection(self):
        if self.selection_mask is not None:
            self.save_state()
            
            self.selection_mask.fill(False)
            self.update_display()
            self.update_selection_info()
            
    def update_selection_info(self):
        if self.selection_mask is None:
            self.selection_info.configure(text="No selection")
        else:
            num_selected = np.sum(self.selection_mask)
            if num_selected == 0:
                self.selection_info.configure(text="No selection")
            else:
                self.selection_info.configure(text=f"{num_selected} pixels selected")
        
    def canvas_click(self, event):
        if self.image_array is None:
            return
            
        if self.current_tool == "brush":
            self.is_painting = True
            self.save_state()
            self.paint_pixel(event.x, event.y)
        elif self.current_tool == "pipette":
            self.pipette_pixel(event.x, event.y)
        elif self.current_tool == "single_select":
            self.single_pixel_select(event.x, event.y)
        elif self.current_tool == "rect_select":
            self.start_rect_select(event.x, event.y)
        elif self.current_tool == "magic_wand":
            self.magic_wand_select(event.x, event.y)
        elif self.current_tool == "object_select":
            self.object_select_improved(event.x, event.y)
            
    def canvas_drag(self, event):
        if self.is_painting and self.current_tool == "brush":
            self.paint_pixel(event.x, event.y)
        elif self.current_tool == "rect_select" and self.is_selecting:
            self.update_rect_select(event.x, event.y)
            
    def canvas_release(self, event):
        self.is_painting = False
        if self.current_tool == "rect_select" and self.is_selecting:
            self.finish_rect_select(event.x, event.y)
        
    def single_pixel_select(self, canvas_x, canvas_y):
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        
        if pixel_x is not None and pixel_y is not None:
            self.save_state()
            
            self.selection_mask[pixel_y, pixel_x] = not self.selection_mask[pixel_y, pixel_x]
            self.update_display()
            self.update_selection_info()

    def start_rect_select(self, canvas_x, canvas_y):
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        if pixel_x is not None and pixel_y is not None:
            self.is_selecting = True
            self.rect_start_x = pixel_x
            self.rect_start_y = pixel_y
            self.save_state()

    def update_rect_select(self, canvas_x, canvas_y):
        if not self.is_selecting:
            return
            
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        if pixel_x is not None and pixel_y is not None:
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            
            start_canvas_x = self.canvas_offset_x + self.rect_start_x * self.zoom_level
            start_canvas_y = self.canvas_offset_y + self.rect_start_y * self.zoom_level
            end_canvas_x = self.canvas_offset_x + pixel_x * self.zoom_level
            end_canvas_y = self.canvas_offset_y + pixel_y * self.zoom_level
            
            self.selection_rect = self.canvas.create_rectangle(
                start_canvas_x, start_canvas_y, end_canvas_x, end_canvas_y,
                outline="red", width=2, tags="temp_rect"
            )

    def finish_rect_select(self, canvas_x, canvas_y):
        if not self.is_selecting:
            return
            
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        if pixel_x is not None and pixel_y is not None:
            min_x = min(self.rect_start_x, pixel_x)
            max_x = max(self.rect_start_x, pixel_x)
            min_y = min(self.rect_start_y, pixel_y)
            max_y = max(self.rect_start_y, pixel_y)
            
            self.selection_mask[min_y:max_y+1, min_x:max_x+1] = True
            
        self.is_selecting = False
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            
        self.update_display()
        self.update_selection_info()

    def magic_wand_select(self, canvas_x, canvas_y):
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        
        if pixel_x is not None and pixel_y is not None:
            self.save_state()
            
            target_color = self.image_array[pixel_y, pixel_x]
            
            visited = np.zeros_like(self.selection_mask)
            stack = [(pixel_y, pixel_x)]
            
            # 8-directional offsets including diagonals
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            while stack:
                y, x = stack.pop()
                
                if x < 0 or x >= self.original_size[1] or y < 0 or y >= self.original_size[0]:
                    continue
                    
                if visited[y, x]:
                    continue
                    
                current_color = self.image_array[y, x]
                
                if np.array_equal(current_color, target_color):
                    visited[y, x] = True
                    self.selection_mask[y, x] = True
                    
                    for dy, dx in directions:
                        stack.append((y + dy, x + dx))
            
            self.update_display()
            self.update_selection_info()

    def object_select_improved(self, canvas_x, canvas_y):
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        
        if pixel_x is not None and pixel_y is not None:
            self.save_state()
            
            target_color = self.image_array[pixel_y, pixel_x]
            height, width = self.original_size
            
            margin = max(50, min(width, height) // 10)
            
            min_row = max(pixel_y - margin, 0)
            max_row = min(pixel_y + margin + 1, height)
            min_col = max(pixel_x - margin, 0)
            max_col = min(pixel_x + margin + 1, width)
            
            sub_image = self.image_array[min_row:max_row, min_col:max_col]
            local_pixel_y = pixel_y - min_row
            local_pixel_x = pixel_x - min_col
            
            color_mask = np.all(sub_image == target_color, axis=2).astype(np.uint8)
            
            structure = np.ones((3, 3), dtype=np.uint8)
            
            labeled_array, num_features = label(color_mask, structure=structure)
            
            target_label = labeled_array[local_pixel_y, local_pixel_x]
            
            if target_label > 0:
                component_mask = labeled_array == target_label
                
                expanded_mask = self.expand_selection_flood_fill_diagonal(
                    sub_image, component_mask, target_color, tolerance=5
                )
                
                self.selection_mask[min_row:max_row, min_col:max_col] |= expanded_mask
            
            self.update_display()
            self.update_selection_info()

    def expand_selection_flood_fill_diagonal(self, image, initial_mask, target_color, tolerance=5):
        height, width = image.shape[:2]
        expanded_mask = initial_mask.copy()
        
        boundary_pixels = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for y in range(height):
            for x in range(width):
                if initial_mask[y, x]:
                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            not initial_mask[ny, nx]):
                            boundary_pixels.append((y, x))
                            break
        
        for start_y, start_x in boundary_pixels:
            stack = [(start_y, start_x)]
            visited = np.zeros((height, width), dtype=bool)
            
            while stack:
                y, x = stack.pop()
                
                if (x < 0 or x >= width or y < 0 or y >= height or 
                    visited[y, x] or expanded_mask[y, x]):
                    continue
                
                current_color = image[y, x]
                color_diff = np.sqrt(np.sum((current_color.astype(float) - 
                                        target_color.astype(float)) ** 2))
                
                if color_diff <= tolerance:
                    visited[y, x] = True
                    expanded_mask[y, x] = True
                    for dy, dx in directions:
                        stack.append((y + dy, x + dx))
        
        return expanded_mask

    def canvas_to_pixel(self, canvas_x, canvas_y):
        image_x = self.canvas.canvasx(canvas_x) - self.canvas_offset_x
        image_y = self.canvas.canvasy(canvas_y) - self.canvas_offset_y
        
        pixel_x = int(image_x / self.zoom_level)
        pixel_y = int(image_y / self.zoom_level)
        
        if (0 <= pixel_x < self.original_size[1] and 
            0 <= pixel_y < self.original_size[0]):
            return pixel_x, pixel_y
        else:
            return None, None
    
    def paint_pixel(self, canvas_x, canvas_y):
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        
        if pixel_x is not None and pixel_y is not None:
            if self.image_array.shape[2] == 3:
                self.image_array[pixel_y, pixel_x, :3] = self.brush_color
            else:
                self.image_array[pixel_y, pixel_x, :3] = self.brush_color
                self.image_array[pixel_y, pixel_x, 3] = 255
                
            self.update_display()
    
    def pipette_pixel(self, canvas_x, canvas_y):
        pixel_x, pixel_y = self.canvas_to_pixel(canvas_x, canvas_y)
        
        if pixel_x is not None and pixel_y is not None:
            pixel_color = self.image_array[pixel_y, pixel_x]
            
            if len(pixel_color) >= 3:
                self.brush_color = tuple(pixel_color[:3])
                
                hex_color = f"#{pixel_color[0]:02x}{pixel_color[1]:02x}{pixel_color[2]:02x}"
                self.color_display.configure(bg=hex_color)
                self.rgb_label.configure(text=f"RGB: {self.brush_color}")
                
                self.tool_var.set("brush")
                self.change_tool()
    
    def save_state(self):
        if self.image_array is not None:
            state = {
                'image': self.image_array.copy(),
                'selection': self.selection_mask.copy() if self.selection_mask is not None else None
            }
            
            self.undo_stack.append(state)
            
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
            
            self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            current_state = {
                'image': self.image_array.copy(),
                'selection': self.selection_mask.copy() if self.selection_mask is not None else None
            }
            self.redo_stack.append(current_state)
            
            self.undo_stack.pop()
            
            prev_state = self.undo_stack[-1]
            self.image_array = prev_state['image'].copy()
            if prev_state['selection'] is not None:
                self.selection_mask = prev_state['selection'].copy()
            
            self.update_display()
            self.update_selection_info()

    def redo(self):
        if self.redo_stack:
            current_state = {
                'image': self.image_array.copy(),
                'selection': self.selection_mask.copy() if self.selection_mask is not None else None
            }
            self.undo_stack.append(current_state)
            
            redo_state = self.redo_stack.pop()
            self.image_array = redo_state['image'].copy()
            if redo_state['selection'] is not None:
                self.selection_mask = redo_state['selection'].copy()
            
            self.update_display()
            self.update_selection_info()
        
    def update_image_info(self):
        if self.image_array is None:
            self.info_label.configure(text="No image loaded")
        else:
            height, width, channels = self.image_array.shape
            channel_text = "RGB" if channels == 3 else "RGBA"
            info_text = f"Size: {width}x{height}\nChannels: {channel_text}\nData type: {self.image_array.dtype}"
            self.info_label.configure(text=info_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelEditor(root)
    root.mainloop()
