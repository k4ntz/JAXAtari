"""
Amidar Maze Editor (visual)

Run `python .\\scripts\\amidar_maze_generator.py`

Tip: Press F1 inside the app to open the User Guide.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.ttk as ttk

import numpy as np

# Make sure we can import from project src
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

try:
    # Use the original template module for constants and default maze
    from jaxatari.games.amidar_mazes import original as ORIG
except Exception as e:
    messagebox.showerror("Import Error", f"Failed to import original template: {e}")
    raise

# Load constants from the original template
WIDTH = int(ORIG.WIDTH)
HEIGHT = int(ORIG.HEIGHT)
TH_H = int(ORIG.PATH_THICKNESS_HORIZONTAL)  # horizontal thickness (height of a horizontal line)
TH_V = int(ORIG.PATH_THICKNESS_VERTICAL)    # vertical thickness (width of a vertical line)

# Optional template grid from the original's default corners (to help alignment)
try:
    _tmpl = np.asarray(ORIG.PATH_CORNERS)
    TEMPLATE_XS = sorted({int(p[0]) for p in _tmpl.tolist()}) if _tmpl.size else []
    TEMPLATE_YS = sorted({int(p[1]) for p in _tmpl.tolist()}) if _tmpl.size else []
    TEMPLATE_CORNERS = [(int(p[0]), int(p[1])) for p in (_tmpl.tolist() if _tmpl.size else [])]
except Exception:
    TEMPLATE_XS, TEMPLATE_YS = [], []
    TEMPLATE_CORNERS = []

# Original template edges for background rendering
try:
    _tmpl_h = np.asarray(ORIG.HORIZONTAL_PATH_EDGES)
    TEMPLATE_H_EDGES = [
        ((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in (_tmpl_h.tolist() if _tmpl_h.size else [])
    ]
except Exception:
    TEMPLATE_H_EDGES = []
try:
    _tmpl_v = np.asarray(ORIG.VERTICAL_PATH_EDGES)
    TEMPLATE_V_EDGES = [
        ((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in (_tmpl_v.tolist() if _tmpl_v.size else [])
    ]
except Exception:
    TEMPLATE_V_EDGES = []


class MazeEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Amidar Maze Editor")

        # Scale for display (purely visual)
        self.scale = 3  # change if you want bigger/smaller
        self.canvas_w = WIDTH * self.scale
        self.canvas_h = HEIGHT * self.scale

        # State
        self.corners = set()           # {(x, y), ...}
        self.h_edges = set()           # {((x1,y),(x2,y)), ...} with x1 < x2
        self.v_edges = set()           # {((x,y1),(x,y2)), ...} with y1 < y2
        self.tool = tk.StringVar(value="corner")  # "corner" or "connect"
        self.pending_connect = None    # first selected corner in connect mode
        self.snap_radius_world = 3     # click tolerance to pick existing corners (world px)
        self.align_snap_world = 8      # tolerance to snap x/y to existing/template axes
        self.snap_to_axes = tk.BooleanVar(value=True)      # snap to existing corners' x/y
        self.snap_to_template = tk.BooleanVar(value=True if (TEMPLATE_XS or TEMPLATE_YS) else False)  # snap to template grid
        self.show_template = tk.BooleanVar(value=True if (TEMPLATE_H_EDGES or TEMPLATE_V_EDGES) else False)

        # UI
        self._build_ui()

        # Initial draw
        self._redraw()

    def _build_ui(self):
        # Use ttk styling for a more modern look
        style = ttk.Style(self)
        # Prefer a modern Windows theme if available, fallback to 'clam'
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
            else:
                style.theme_use("clam")
        except Exception:
            pass

        # Subtle style tweaks
        style.configure("TLabelFrame", padding=(6, 6))
        style.configure("TButton", padding=(10, 6))
        style.configure("TRadiobutton", padding=(6, 4))
        style.configure("TCheckbutton", padding=(6, 4))

        # Menu bar (Help)
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide (F1)", command=self._show_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        # Key bindings
        self.bind("<F1>", lambda e: self._show_help())

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(top)
        row1.pack(side=tk.TOP, fill=tk.X)
        row2 = ttk.Frame(top)
        row2.pack(side=tk.TOP, fill=tk.X)

        # Row 1: Mode and Actions groups
        mode_frame = ttk.LabelFrame(row1, text="Mode")
        mode_frame.pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Radiobutton(mode_frame, text="Corner", variable=self.tool, value="corner").pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Radiobutton(mode_frame, text="Connect", variable=self.tool, value="connect").pack(side=tk.LEFT, padx=6, pady=4)

        actions_frame = ttk.LabelFrame(row1, text="Actions")
        actions_frame.pack(side=tk.LEFT, padx=12, pady=4)
        ttk.Button(actions_frame, text="Clear", command=self._on_clear).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(actions_frame, text="Connect all", command=self._connect_all).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(actions_frame, text="Export → File", command=self._export_file).pack(side=tk.LEFT, padx=6, pady=6)

        # Row 2: Template and Snapping groups
        template_frame = ttk.LabelFrame(row2, text="Template")
        template_frame.pack(side=tk.LEFT, padx=6, pady=2)
        ttk.Checkbutton(template_frame, text="Show", variable=self.show_template, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Button(template_frame, text="Add corners", command=self._add_template_corners).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Checkbutton(template_frame, text="Snap grid", variable=self.snap_to_template, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)

        snapping_frame = ttk.LabelFrame(row2, text="Snapping")
        snapping_frame.pack(side=tk.LEFT, padx=12, pady=2)
        ttk.Checkbutton(snapping_frame, text="Snap axes", variable=self.snap_to_axes, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)

        self.status = ttk.Label(self, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(
            self,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#1e1e1e",
            highlightthickness=1,
            highlightbackground="#666666",
            highlightcolor="#666666",
            bd=1,
            relief="solid",
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_motion)

    # --------------------- Help/Manual ---------------------
    def _clear_selection(self):
        # Clear any pending connect selection
        self.pending_connect = None
        self.status.config(text="")

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Amidar Maze Editor\nCreate and export Amidar-style mazes for JAXAtari.",
        )

    def _show_help(self):
        # Create a simple scrollable help window
        win = tk.Toplevel(self)
        win.title("User Guide")
        win.geometry("720x520")
        win.transient(self)
        win.grab_set()

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(frm, wrap="word")
        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        guide = f"""
Amidar Maze Editor — User Guide

Modes
- Corner mode: Left-click toggles a corner on/off at the snapped position.
- Connect mode: Left-click a corner to start, then left-click an aligned corner to toggle the path segments between them. Right-click cancels the selection.

Snapping and Template
- Snap axes: Snap to existing corners' X/Y to keep lines straight.
- Snap grid: Snap to the original template's grid (from amidar_mazes.original).
- Show template: Toggle background rendering of the original maze.
- Add corners: Add all template corners to your current set.

Actions
- Connect all: Connect all adjacent corners along rows and columns where valid.
- Clear: Remove all corners and edges.
- Export → File: Writes a Python module with JAX arrays into src/jaxatari/games/amidar_mazes/maze_N.py.

Importing in code
- After exporting: from jaxatari.games.amidar_mazes.maze_N import PATH_CORNERS, HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES
- The original maze is available at jaxatari.games.amidar_mazes.original

Tips
- Use the template grid and snap axes to keep everything aligned.
- The status bar shows coordinates and snapping hints under the cursor.

Validation (automatic)
- Edges are axis-aligned; intersections are only allowed at corners.
- Export blocks if there are dead-ends (corners with degree 1).
"""
        txt.insert("1.0", guide)
        txt.configure(state="disabled")

    # --------------------- Coordinate helpers ---------------------
    def _world_to_screen(self, x, y):
        return x * self.scale, y * self.scale

    def _screen_to_world(self, sx, sy):
        # nearest integer pixel
        x = int(round(sx / self.scale))
        y = int(round(sy / self.scale))
        # clamp
        x = max(0, min(WIDTH - 1, x))
        y = max(0, min(HEIGHT - 1, y))
        return x, y

    def _nearest_corner(self, x, y):
        # Find nearest corner within radius
        best = None
        br = self.snap_radius_world
        for cx, cy in self.corners:
            if abs(cx - x) <= br and abs(cy - y) <= br:
                if best is None:
                    best = (cx, cy)
                else:
                    # pick closer in L1 distance
                    if abs(cx - x) + abs(cy - y) < abs(best[0] - x) + abs(best[1] - y):
                        best = (cx, cy)
        return best

    def _axis_candidates(self):
        xs = set()
        ys = set()
        if self.snap_to_axes.get():
            for cx, cy in self.corners:
                xs.add(cx)
                ys.add(cy)
        if self.snap_to_template.get():
            xs.update(TEMPLATE_XS)
            ys.update(TEMPLATE_YS)
        return sorted(xs), sorted(ys)

    def _snap_point(self, x, y):
        # Snap x/y to nearest candidate axes within tolerance
        xs, ys = self._axis_candidates()
        snapped_x = x
        snapped_y = y
        used_x = False
        used_y = False
        if xs:
            nearest_x = min(xs, key=lambda v: abs(v - x))
            if abs(nearest_x - x) <= self.align_snap_world:
                snapped_x = nearest_x
                used_x = True
        if ys:
            nearest_y = min(ys, key=lambda v: abs(v - y))
            if abs(nearest_y - y) <= self.align_snap_world:
                snapped_y = nearest_y
                used_y = True
        return snapped_x, snapped_y, used_x, used_y

    # --------------------- Corner operations ---------------------
    def _add_corner(self, x, y):
        if (x, y) in self.corners:
            return
        self.corners.add((x, y))
        self._redraw()

    def _remove_corner(self, x, y):
        if (x, y) not in self.corners:
            return
        # remove edges that use this corner
        self.h_edges = {e for e in self.h_edges if e[0] != (x, y) and e[1] != (x, y)}
        self.v_edges = {e for e in self.v_edges if e[0] != (x, y) and e[1] != (x, y)}
        self.corners.remove((x, y))
        # reset pending connect if it referenced this corner
        if self.pending_connect == (x, y):
            self.pending_connect = None
        self._redraw()

    def _add_template_corners(self):
        if not TEMPLATE_CORNERS:
            messagebox.showwarning("No template", "No template corners available.")
            return
        before = len(self.corners)
        self.corners.update(TEMPLATE_CORNERS)
        added = len(self.corners) - before
        self._redraw()
        if added:
            self.status.config(text=f"Added {added} template corners.")

    def _ensure_connect(self, a, b):
        # Like _toggle_connect but only adds missing segments; never removes
        if not (a[0] == b[0] or a[1] == b[1]):
            return
        pts = self._sorted_line_corners_between(a, b)
        if len(pts) < 2:
            return
        if a[1] == b[1]:
            y = a[1]
            for i in range(len(pts) - 1):
                x1, _ = pts[i]
                x2, _ = pts[i + 1]
                if x1 > x2:
                    x1, x2 = x2, x1
                seg = ((x1, y), (x2, y))
                if seg not in self.h_edges:
                    if not self._intersections_valid_for_horizontal(y, x1, x2):
                        continue
                    self.h_edges.add(seg)
        else:
            x = a[0]
            for i in range(len(pts) - 1):
                _, y1 = pts[i]
                _, y2 = pts[i + 1]
                if y1 > y2:
                    y1, y2 = y2, y1
                seg = ((x, y1), (x, y2))
                if seg not in self.v_edges:
                    if not self._intersections_valid_for_vertical(x, y1, y2):
                        continue
                    self.v_edges.add(seg)

    def _connect_all(self):
        # For each row and column, connect between min and max across existing corners
        if not self.corners:
            return
        by_y = {}
        by_x = {}
        for (cx, cy) in self.corners:
            by_y.setdefault(cy, []).append(cx)
            by_x.setdefault(cx, []).append(cy)
        # Rows
        for y, xs in by_y.items():
            if len(xs) >= 2:
                xs_sorted = sorted(xs)
                a = (xs_sorted[0], y)
                b = (xs_sorted[-1], y)
                self._ensure_connect(a, b)
        # Cols
        for x, ys in by_x.items():
            if len(ys) >= 2:
                ys_sorted = sorted(ys)
                a = (x, ys_sorted[0])
                b = (x, ys_sorted[-1])
                self._ensure_connect(a, b)
        self._redraw()

    # Removed disconnect-all per request

    # --------------------- Edge operations ---------------------
    def _sorted_line_corners_between(self, a, b):
        # Return list of corners on the same row or column between a and b (inclusive), sorted
        (x1, y1), (x2, y2) = a, b
        if x1 == x2:
            # vertical
            xs = x1
            low, high = sorted([y1, y2])
            pts = [(x, y) for (x, y) in self.corners if x == xs and low <= y <= high]
            pts.sort(key=lambda t: t[1])
            return pts
        elif y1 == y2:
            # horizontal
            ys = y1
            low, high = sorted([x1, x2])
            pts = [(x, y) for (x, y) in self.corners if y == ys and low <= x <= high]
            pts.sort(key=lambda t: t[0])
            return pts
        else:
            return []

    def _intersections_valid_for_horizontal(self, y, x1, x2):
        # New horizontal segment (x1,y)->(x2,y), with x1 < x2
        for (vx, vy1), (_, vy2) in self.v_edges:
            if vy1 < y < vy2 and x1 < vx < x2:
                # intersection at (vx, y) must be an existing corner
                if (vx, y) not in self.corners:
                    return False
        return True

    def _intersections_valid_for_vertical(self, x, y1, y2):
        # New vertical segment (x,y1)->(x,y2), with y1 < y2
        for (hx1, hy), (hx2, _) in self.h_edges:
            if hx1 < x < hx2 and y1 < hy < y2:
                # intersection at (x, hy) must be an existing corner
                if (x, hy) not in self.corners:
                    return False
        return True

    def _toggle_connect(self, a, b):
        # Only allow same row/column
        if not (a[0] == b[0] or a[1] == b[1]):
            messagebox.showwarning("Not aligned", "Corners must be aligned horizontally or vertically.")
            return

        pts = self._sorted_line_corners_between(a, b)
        if len(pts) < 2:
            return

        # Build all adjacent segments across the span
        segments = []
        if a[1] == b[1]:
            # horizontal
            y = a[1]
            for i in range(len(pts) - 1):
                x1, _ = pts[i]
                x2, _ = pts[i + 1]
                if x1 > x2:
                    x1, x2 = x2, x1
                segments.append(("H", (x1, y), (x2, y)))
        else:
            # vertical
            x = a[0]
            for i in range(len(pts) - 1):
                _, y1 = pts[i]
                _, y2 = pts[i + 1]
                if y1 > y2:
                    y1, y2 = y2, y1
                segments.append(("V", (x, y1), (x, y2)))

        # Determine if all segments already exist => then we'll remove, else add (but validate first)
        all_exist = True
        for t, p1, p2 in segments:
            if t == "H":
                seg = (p1, p2)
                if seg not in self.h_edges:
                    all_exist = False
                    break
            else:
                seg = (p1, p2)
                if seg not in self.v_edges:
                    all_exist = False
                    break

        if all_exist:
            # remove them
            for t, p1, p2 in segments:
                if t == "H":
                    self.h_edges.discard((p1, p2))
                else:
                    self.v_edges.discard((p1, p2))
        else:
            # try to add all (validate intersections)
            # Validate each proposed segment
            for t, p1, p2 in segments:
                if t == "H":
                    (x1, y), (x2, _) = p1, p2
                    if not self._intersections_valid_for_horizontal(y, x1, x2):
                        messagebox.showerror("Invalid crossing", "Horizontal segment would cross a vertical segment at a non-corner.")
                        return
                else:
                    (x, y1), (_, y2) = p1, p2
                    if not self._intersections_valid_for_vertical(x, y1, y2):
                        messagebox.showerror("Invalid crossing", "Vertical segment would cross a horizontal segment at a non-corner.")
                        return
            # Passed validation; add missing ones
            for t, p1, p2 in segments:
                if t == "H":
                    self.h_edges.add((p1, p2))
                else:
                    self.v_edges.add((p1, p2))

        self._redraw()

    # --------------------- Events ---------------------
    def _on_left_click(self, ev):
        x, y = self._screen_to_world(ev.x, ev.y)
        # Apply snapping for easier aligned placement
        x, y, _, _ = self._snap_point(x, y)
        if self.tool.get() == "corner":
            # Toggle corner on left click
            near = self._nearest_corner(x, y)
            if near is None:
                self._add_corner(x, y)
            else:
                self._remove_corner(*near)
                self.pending_connect = None
                # _remove_corner calls _redraw()
        else:
            # connect mode
            near = self._nearest_corner(x, y)
            if near is None:
                return
            if self.pending_connect is None:
                self.pending_connect = near
                self._redraw()
            else:
                a = self.pending_connect
                b = near
                self.pending_connect = None
                if a != b:
                    self._toggle_connect(a, b)
                else:
                    self._redraw()

    def _on_right_click(self, ev):
        # No-op in corner mode; in connect mode, right-click clears selection
        if self.tool.get() == "connect":
            self.pending_connect = None
            self._redraw()

    def _on_motion(self, ev):
        x, y = self._screen_to_world(ev.x, ev.y)
        sx, sy, use_x, use_y = self._snap_point(x, y)
        self.status.config(text=f"Tool: {self.tool.get()} | x={x}→{sx if use_x else x}, y={y}→{sy if use_y else y} | corners={len(self.corners)} | horizontal edges={len(self.h_edges)} | vertical edges={len(self.v_edges)}")
        self._draw_guides(sx, sy, use_x, use_y)

    def _on_clear(self):
        if messagebox.askyesno("Clear", "Remove all corners and edges?"):
            self.corners.clear()
            self.h_edges.clear()
            self.v_edges.clear()
            self.pending_connect = None
            self._redraw()

    # --------------------- Draw ---------------------
    def _redraw(self):
        self.canvas.delete("all")

        # Draw template (original) maze background if enabled
        if self.show_template.get():
            bg_color = "#9a9a9a" if self.snap_to_template.get() else "#555555"
            for (x1, y), (x2, _) in TEMPLATE_H_EDGES:
                sx1, sy = self._world_to_screen(x1, y)
                sx2, _ = self._world_to_screen(x2, y)
                self.canvas.create_line(sx1, sy, sx2, sy, fill=bg_color, width=max(1, TH_H * self.scale))
            for (x, y1), (_, y2) in TEMPLATE_V_EDGES:
                sx, sy1 = self._world_to_screen(x, y1)
                _, sy2 = self._world_to_screen(x, y2)
                self.canvas.create_line(sx, sy1, sx, sy2, fill=bg_color, width=max(1, TH_V * self.scale))

        # Draw edges first
        # Horizontal: thickness in display = TH_H * scale (height of the horizontal bar)
        for (x1, y), (x2, _) in sorted(self.h_edges):
            sx1, sy = self._world_to_screen(x1, y)
            sx2, _ = self._world_to_screen(x2, y)
            self.canvas.create_line(sx1, sy, sx2, sy, fill="#5bb05b", width=max(1, TH_H * self.scale), capstyle=tk.ROUND)

        # Vertical: thickness in display = TH_V * scale (width of the vertical bar)
        for (x, y1), (_, y2) in sorted(self.v_edges):
            sx, sy1 = self._world_to_screen(x, y1)
            _, sy2 = self._world_to_screen(x, y2)
            self.canvas.create_line(sx, sy1, sx, sy2, fill="#5bb05b", width=max(1, TH_V * self.scale), capstyle=tk.ROUND)

        # Draw corners
        for (cx, cy) in self.corners:
            sx, sy = self._world_to_screen(cx, cy)
            r = max(2, int(2 * self.scale))  # draw a small square
            color = "#ffcc00" if self.pending_connect == (cx, cy) else "#ff5555"
            self.canvas.create_rectangle(sx - r, sy - r, sx + r, sy + r, fill=color, outline="black", width=1)

        # Remove any lingering guides; they'll be redrawn on next motion
        self.canvas.delete("guide")
        # Draw canvas outline to ensure visible border
        self.canvas.create_rectangle(1, 1, self.canvas_w - 1, self.canvas_h - 1, outline="#888888", width=2, tags=("frame",))

    def _draw_guides(self, x, y, use_x, use_y):
        # Draw alignment guides at snapped axes
        self.canvas.delete("guide")
        if use_x:
            sx, _ = self._world_to_screen(x, 0)
            self.canvas.create_line(sx, 0, sx, self.canvas_h, fill="#5a5a5a", dash=(4, 3), tags=("guide",))
        if use_y:
            _, sy = self._world_to_screen(0, y)
            self.canvas.create_line(0, sy, self.canvas_w, sy, fill="#5a5a5a", dash=(4, 3), tags=("guide",))

    # --------------------- Export ---------------------
    def _validate_for_export(self):
        # No illegal crossings (already enforced)
        # Check degrees: corners should not have degree 1 (avoid dead-ends)
        deg = {c: 0 for c in self.corners}
        for (a, b) in self.h_edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        for (a, b) in self.v_edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        bad = [c for c, d in deg.items() if d == 1]
        if bad:
            return False, f"Dead-end at corners: {bad[:5]}{'...' if len(bad) > 5 else ''}"
        return True, "OK"

    def _assemble_arrays(self):
        # Sort corners by (y, then x) for stable output
        corners_sorted = sorted(self.corners, key=lambda t: (t[1], t[0]))
        corners_arr = np.array(corners_sorted, dtype=np.int32)

        # Prepare edges arrays with canonical direction
        horiz_sorted = []
        for (x1, y), (x2, _) in sorted(self.h_edges):
            if x1 > x2:
                x1, x2 = x2, x1
            horiz_sorted.append([[x1, y], [x2, y]])

        vert_sorted = []
        for (x, y1), (_, y2) in sorted(self.v_edges):
            if y1 > y2:
                y1, y2 = y2, y1
            vert_sorted.append([[x, y1], [x, y2]])

        h_arr = np.array(horiz_sorted, dtype=np.int32) if horiz_sorted else np.zeros((0, 2, 2), dtype=np.int32)
        v_arr = np.array(vert_sorted, dtype=np.int32) if vert_sorted else np.zeros((0, 2, 2), dtype=np.int32)
        return corners_arr, h_arr, v_arr

    def _export_file(self):
        ok, msg = self._validate_for_export()
        if not ok:
            messagebox.showerror("Invalid maze", msg)
            return
        corners_arr, h_arr, v_arr = self._assemble_arrays()

        # Save into organized folder: src/jaxatari/games/amidar_mazes
        mazes_dir = os.path.join(PROJECT_SRC, "jaxatari", "games", "amidar_mazes")
        try:
            os.makedirs(mazes_dir, exist_ok=True)
            # ensure it's a package
            init_path = os.path.join(mazes_dir, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write("# Amidar maze variants\n")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to create mazes folder: {e}")
            return

        # Determine next incremental filename: maze_0.py, maze_1.py, ...
        existing = set()
        try:
            for name in os.listdir(mazes_dir):
                if name.startswith("maze_") and name.endswith(".py"):
                    num_part = name[len("maze_"):-3]
                    if num_part.isdigit():
                        existing.add(int(num_part))
        except Exception:
            pass
        n = 0
        while n in existing:
            n += 1
        filename = f"maze_{n}.py"
        path = os.path.join(mazes_dir, filename)

        content = []
        content.append("# Auto-generated Amidar maze arrays (JAX)")
        content.append("import jax.numpy as jnp")
        content.append("")
        content.append(f"WIDTH = {WIDTH}")
        content.append(f"HEIGHT = {HEIGHT}")
        content.append(f"PATH_THICKNESS_HORIZONTAL = {TH_H}")
        content.append(f"PATH_THICKNESS_VERTICAL = {TH_V}")
        content.append("")
        content.append(f"PATH_CORNERS = jnp.array({corners_arr.tolist()}, dtype=jnp.int32)")
        content.append(f"HORIZONTAL_PATH_EDGES = jnp.array({h_arr.tolist()}, dtype=jnp.int32)")
        content.append(f"VERTICAL_PATH_EDGES = jnp.array({v_arr.tolist()}, dtype=jnp.int32)")
        content.append("")
        content.append("# Import these from this file where needed instead of modifying jax_amidar.py.")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            module_name = f"jaxatari.games.amidar_mazes.{os.path.splitext(filename)[0]}"
            messagebox.showinfo(
                "Saved",
                (
                    f"Saved to {path}\n\n"
                ),
            )
        except Exception as e:
            messagebox.showerror("Save error", str(e))


if __name__ == "__main__":
    app = MazeEditor()
    app.mainloop()