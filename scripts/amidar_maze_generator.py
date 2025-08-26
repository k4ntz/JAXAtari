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
        self.tool = tk.StringVar(value="corner")  # "corner" or "connect" or "player" or "enemy"
        # Entity placement
        self.player_pos = None
        self.enemy_positions = []
        self.pending_connect = None    # first selected corner in connect mode
        self.snap_radius_world = 3     # click tolerance to pick existing corners (world px)
        self.align_snap_world = 8      # tolerance to snap x/y to existing/template axes
        self.snap_to_axes = tk.BooleanVar(value=True)      # snap to existing corners' x/y
        self.snap_to_template = tk.BooleanVar(value=False)  # snap to template grid
        self.show_template = tk.BooleanVar(value=True if (TEMPLATE_H_EDGES or TEMPLATE_V_EDGES) else False)
        # Perimeter overlay toggle
        self.show_perimeter = tk.BooleanVar(value=True)
        # Rectangles overlay toggle
        self.show_rectangles = tk.BooleanVar(value=False)

        # Corner-rectangle manual selection state
        self.corner_select_active = False
        self.corner_rectangle_overrides = []
        self._rect_bounds_cache = None  # np.ndarray or None
        self._rect_masks_cache = None   # np.ndarray or None

        # UI
        self._build_ui()

        # Initial draw
        self._redraw()

    # --------------------- Internal state helpers ---------------------
    def _invalidate_rect_cache(self, graph_changed: bool = False):
        """Invalidate cached rectangle data; optionally reset manual selection if topology changed."""
        self._rect_bounds_cache = None
        self._rect_masks_cache = None
        if graph_changed and self.corner_select_active:
            # Reset manual selection as rectangles likely changed
            self.corner_select_active = False
            self.corner_rectangle_overrides = []
            try:
                self.status.config(text="Corner-rectangle selection reset due to maze changes.")
            except Exception:
                pass

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
        menubar.add_command(label="Export", command=self._export_file)
        self.config(menu=menubar)

        # Key bindings
        self.bind("<F1>", lambda e: self._show_help())
        self.bind("<Control-s>", lambda e: self._export_file())

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
        ttk.Radiobutton(mode_frame, text="Player", variable=self.tool, value="player").pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Radiobutton(mode_frame, text="Enemy", variable=self.tool, value="enemy").pack(side=tk.LEFT, padx=6, pady=4)

        # Row 2: Template and Snapping groups
        template_frame = ttk.LabelFrame(row1, text="Template")
        template_frame.pack(side=tk.LEFT, padx=6, pady=2)
        ttk.Checkbutton(template_frame, text="Show", variable=self.show_template, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Checkbutton(template_frame, text="Snap template", variable=self.snap_to_template, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Button(template_frame, text="Add corners", command=self._add_template_corners).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(template_frame, text="Add player & enemies", command=self._use_original_positions).pack(side=tk.LEFT, padx=8)

        tool_frame = ttk.LabelFrame(row2, text="Tools")
        tool_frame.pack(side=tk.LEFT, padx=12, pady=2)
        ttk.Checkbutton(tool_frame, text="Snap axis", variable=self.snap_to_axes, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Checkbutton(tool_frame, text="Show perimeter", variable=self.show_perimeter, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Checkbutton(tool_frame, text="Show rectangles", variable=self.show_rectangles, command=self._redraw).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Button(tool_frame, text="Connect all", command=self._connect_all).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(tool_frame, text="Clear", command=self._on_clear).pack(side=tk.LEFT, padx=6, pady=6)

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

    # --------------------- Precompute helpers (NumPy versions) ---------------------
    # These mirror the logic in src/jaxatari/games/jax_amidar.py for path-derived constants,
    # adapted to NumPy for offline computation at export time.

    @staticmethod
    def _build_edge_index_and_neighbors(h_arr: np.ndarray, v_arr: np.ndarray):
        """
        Build mapping from edge -> index (PATH_EDGES indexing: horizontals first, then verticals),
        and neighbor lookups per corner: right/left/up/down immediate neighbors.

        h_arr shape: (H, 2, 2) [[x1,y], [x2,y]] with x1 < x2
        v_arr shape: (V, 2, 2) [[x,y1], [x,y2]] with y1 < y2
        """
        edge_index = {}
        neighbors_right = {}
        neighbors_left = {}
        neighbors_up = {}
        neighbors_down = {}

        # Horizontal edges first
        for i, e in enumerate(h_arr.tolist()):
            (x1, y1), (x2, y2) = e
            # canonical
            if (x2, y2) < (x1, y1):
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
            key = ((x1, y1), (x2, y2))
            edge_index[key] = i
            # neighbors
            neighbors_right[(x1, y1)] = (x2, y2)
            neighbors_left[(x2, y2)] = (x1, y1)

        # Vertical edges next
        offset = len(h_arr)
        for j, e in enumerate(v_arr.tolist()):
            (x1, y1), (x2, y2) = e
            # canonical (y1 < y2)
            if (x2, y2) < (x1, y1):
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
            key = ((x1, y1), (x2, y2))
            edge_index[key] = offset + j
            # neighbors
            neighbors_down[(x1, y1)] = (x2, y2)
            neighbors_up[(x2, y2)] = (x1, y1)

        return edge_index, neighbors_right, neighbors_left, neighbors_up, neighbors_down

    @staticmethod
    def _split_edges_at_corners(corners_arr: np.ndarray, h_arr: np.ndarray, v_arr: np.ndarray):
        """Return (h_split, v_split) where any edge spanning over intermediate corners
        is split into sub-segments that start/end at each corner.

        h_arr: (H,2,2) [[x1,y],[x2,y]]; v_arr: (V,2,2) [[x,y1],[x,y2]]
        """
        corners = [tuple(map(int, p)) for p in corners_arr.tolist()] if corners_arr.size else []
        corners_by_y = {}
        corners_by_x = {}
        for cx, cy in corners:
            corners_by_y.setdefault(int(cy), []).append(int(cx))
            corners_by_x.setdefault(int(cx), []).append(int(cy))
        for y in corners_by_y:
            corners_by_y[y].sort()
        for x in corners_by_x:
            corners_by_x[x].sort()

        def _split_h(edges: np.ndarray):
            if edges is None or len(edges) == 0:
                return np.zeros((0, 2, 2), dtype=np.int32)
            out = []
            for (x1, y1), (x2, y2) in edges.tolist():
                y = int(y1)
                xa, xb = int(min(x1, x2)), int(max(x1, x2))
                xs = [xa]
                for cx in corners_by_y.get(y, []):
                    if xa < cx < xb:
                        xs.append(int(cx))
                xs.append(xb)
                for i in range(len(xs) - 1):
                    a, b = xs[i], xs[i + 1]
                    if b > a:
                        out.append([[a, y], [b, y]])
            return np.asarray(out, dtype=np.int32) if out else np.zeros((0, 2, 2), dtype=np.int32)

        def _split_v(edges: np.ndarray):
            if edges is None or len(edges) == 0:
                return np.zeros((0, 2, 2), dtype=np.int32)
            out = []
            for (x1, y1), (x2, y2) in edges.tolist():
                x = int(x1)
                ya, yb = int(min(y1, y2)), int(max(y1, y2))
                ys = [ya]
                for cy in corners_by_x.get(x, []):
                    if ya < cy < yb:
                        ys.append(int(cy))
                ys.append(yb)
                for i in range(len(ys) - 1):
                    a, b = ys[i], ys[i + 1]
                    if b > a:
                        out.append([[x, a], [x, b]])
            return np.asarray(out, dtype=np.int32) if out else np.zeros((0, 2, 2), dtype=np.int32)

        return _split_h(h_arr), _split_v(v_arr)

    @staticmethod
    def _edge_index_lookup(edge_index, a, b):
        """Return PATH_EDGES index for edge between corners a and b using canonical ordering."""
        (x1, y1), (x2, y2) = a, b
        if y1 == y2:  # horizontal
            if x2 < x1:
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        else:  # vertical
            if y2 < y1:
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        return edge_index[((x1, y1), (x2, y2))]

    @staticmethod
    def _compute_rectangles(corners_arr: np.ndarray, h_arr: np.ndarray, v_arr: np.ndarray):
        """
        Compute rectangle edge masks by traversing edges in the order:
        Down until Right exists -> Right until Up exists -> Up until Left exists -> Left until back at start.

        Returns (rect_masks: (R, E) int32, rect_bounds: (R,4) int32).
        """
        # 1) Pre-split edges at intermediate corners so every segment starts/ends at a corner.
        h_arr, v_arr = MazeEditor._split_edges_at_corners(corners_arr, h_arr, v_arr)

        num_h = int(len(h_arr))
        num_v = int(len(v_arr))
        num_edges = num_h + num_v

        # Fast directional lookups
        def find_right_from(c):
            x, y = c
            if num_h == 0:
                return None
            starts = (h_arr[:, 0, 0] == x) & (h_arr[:, 0, 1] == y)
            idxs = np.nonzero(starts)[0]
            return int(idxs[0]) if idxs.size else None

        def find_left_to(c):
            x, y = c
            if num_h == 0:
                return None
            ends = (h_arr[:, 1, 0] == x) & (h_arr[:, 1, 1] == y)
            idxs = np.nonzero(ends)[0]
            return int(idxs[0]) if idxs.size else None

        def find_down_from(c):
            x, y = c
            if num_v == 0:
                return None
            starts = (v_arr[:, 0, 0] == x) & (v_arr[:, 0, 1] == y)
            idxs = np.nonzero(starts)[0]
            return int(idxs[0]) if idxs.size else None

        def find_up_to(c):
            x, y = c
            if num_v == 0:
                return None
            ends = (v_arr[:, 1, 0] == x) & (v_arr[:, 1, 1] == y)
            idxs = np.nonzero(ends)[0]
            return int(idxs[0]) if idxs.size else None

        rect_masks = []
        rect_bounds = []
        seen = set()

        corners = [tuple(map(int, p)) for p in corners_arr.tolist()] if corners_arr.size else []
        for c_start in corners:
            mask = np.zeros((num_edges,), dtype=np.int32)

            # DOWN: keep going until a right edge exists
            c = c_start
            while True:
                j = find_down_from(c)
                if j is None:
                    mask = None
                    break
                mask[num_h + j] = 1
                c = (int(v_arr[j, 1, 0]), int(v_arr[j, 1, 1]))
                if find_right_from(c) is not None:
                    break

            if mask is None:
                continue

            # RIGHT: keep going until an up edge exists
            while True:
                i = find_right_from(c)
                if i is None:
                    mask = None
                    break
                mask[i] = 1
                c = (int(h_arr[i, 1, 0]), int(h_arr[i, 1, 1]))
                if find_up_to(c) is not None:
                    break

            if mask is None:
                continue

            # UP: keep going until a left edge exists
            while True:
                j = find_up_to(c)
                if j is None:
                    mask = None
                    break
                mask[num_h + j] = 1
                c = (int(v_arr[j, 0, 0]), int(v_arr[j, 0, 1]))
                if find_left_to(c) is not None:
                    break

            if mask is None:
                continue

            # LEFT: keep going until back at start
            while True:
                i = find_left_to(c)
                if i is None:
                    mask = None
                    break
                mask[i] = 1
                c = (int(h_arr[i, 0, 0]), int(h_arr[i, 0, 1]))
                if c == c_start:
                    break

            if mask is None:
                continue

            # Bounds from all used edges
            if not mask.any():
                continue
            # Collect coordinates of edges in mask
            xs = []
            ys = []
            # horizontals
            if num_h:
                hi = np.nonzero(mask[:num_h])[0]
                if hi.size:
                    segs = h_arr[hi]
                    xs.extend(segs[:, :, 0].ravel().tolist())
                    ys.extend(segs[:, :, 1].ravel().tolist())
            # verticals
            if num_v:
                vi = np.nonzero(mask[num_h:])[0]
                if vi.size:
                    segs = v_arr[vi]
                    xs.extend(segs[:, :, 0].ravel().tolist())
                    ys.extend(segs[:, :, 1].ravel().tolist())
            if not xs:
                continue
            b = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            key = tuple(mask.tolist())
            if key in seen:
                continue
            seen.add(key)
            rect_masks.append(mask)
            rect_bounds.append(list(b))

        if rect_masks:
            rect_masks = np.stack(rect_masks, axis=0)
            rect_bounds = np.asarray(rect_bounds, dtype=np.int32)
        else:
            rect_masks = np.zeros((0, int(num_edges)), dtype=np.int32)
            rect_bounds = np.zeros((0, 4), dtype=np.int32)

        return rect_masks, rect_bounds

    @staticmethod
    def _compute_corner_rectangles(rect_bounds: np.ndarray):
        """Return indices of the four corner-most rectangles (top-left, top-right, bottom-left, bottom-right)."""
        if rect_bounds.size == 0:
            return np.zeros((4,), dtype=np.int32)

        min_x = int(np.min(rect_bounds[:, 0]))
        min_y = int(np.min(rect_bounds[:, 1]))
        max_x = int(np.max(rect_bounds[:, 2]))
        max_y = int(np.max(rect_bounds[:, 3]))

        idxs = []
        # top-left: min_x & min_y
        for i, (rx1, ry1, rx2, ry2) in enumerate(rect_bounds.tolist()):
            if rx1 == min_x and ry1 == min_y:
                idxs.append(i)
                break
        # top-right: max_x & min_y
        for i, (rx1, ry1, rx2, ry2) in enumerate(rect_bounds.tolist()):
            if rx2 == max_x and ry1 == min_y:
                idxs.append(i)
                break
        # bottom-left: min_x & max_y
        for i, (rx1, ry1, rx2, ry2) in enumerate(rect_bounds.tolist()):
            if rx1 == min_x and ry2 == max_y:
                idxs.append(i)
                break
        # bottom-right: max_x & max_y
        for i, (rx1, ry1, rx2, ry2) in enumerate(rect_bounds.tolist()):
            if rx2 == max_x and ry2 == max_y:
                idxs.append(i)
                break

        # Pad/truncate to exactly 4 entries like jnp.nonzero(..., size=4)
        if len(idxs) < 4:
            idxs += [0] * (4 - len(idxs))
        elif len(idxs) > 4:
            idxs = idxs[:4]
        return np.asarray(idxs, dtype=np.int32)

    def _compute_perimeter_edges_from_edges(self, h_arr: np.ndarray, v_arr: np.ndarray):
        """Compute outer perimeter edges using a raster flood-fill on the union of path bars.
        Returns list of edges as ('h', y, x1, x2) or ('v', x, y1, y2) at centerline coordinates.
        """
        H = int(HEIGHT)
        W = int(WIDTH)
        if (h_arr is None or len(h_arr) == 0) and (v_arr is None or len(v_arr) == 0):
            return []

        path = np.zeros((H, W), dtype=np.uint8)
        # Draw horizontal bars with thickness TH_H
        if h_arr is not None and len(h_arr) > 0:
            for (x1, y), (x2, _) in h_arr.tolist():
                x1 = int(min(x1, x2)); x2 = int(max(x1, x2))
                y = int(y)
                y0 = max(0, y - TH_H // 2)
                y1 = min(H - 1, y0 + TH_H - 1)
                x0 = max(0, x1)
                x1b = min(W - 1, x2)
                if y1 >= y0 and x1b >= x0:
                    path[y0:y1+1, x0:x1b+1] = 1
        # Draw vertical bars with thickness TH_V
        if v_arr is not None and len(v_arr) > 0:
            for (x, y1), (_, y2) in v_arr.tolist():
                x = int(x)
                y1 = int(min(y1, y2)); y2 = int(max(y1, y2))
                x0 = max(0, x - TH_V // 2)
                x1 = min(W - 1, x0 + TH_V - 1)
                y0 = max(0, y1)
                y1b = min(H - 1, y2)
                if y1b >= y0 and x1 >= x0:
                    path[y0:y1b+1, x0:x1+1] = 1

        # Flood-fill outside on padded background
        from collections import deque
        H2, W2 = H + 2, W + 2
        pad = np.zeros((H2, W2), dtype=np.uint8)
        pad[1:H+1, 1:W+1] = path
        outside = np.zeros_like(pad, dtype=np.uint8)
        dq = deque([(0, 0)])
        outside[0, 0] = 1
        while dq:
            y, x = dq.popleft()
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < H2 and 0 <= nx < W2 and outside[ny, nx] == 0 and pad[ny, nx] == 0:
                    outside[ny, nx] = 1
                    dq.append((ny, nx))

        edges = []
        # Check each horizontal centerline for adjacency to outside
        if h_arr is not None and len(h_arr) > 0:
            for (x1, y), (x2, _) in h_arr.tolist():
                x1 = int(min(x1, x2)); x2 = int(max(x1, x2))
                y = int(y)
                y0 = max(0, y - TH_H // 2)
                y1 = min(H - 1, y0 + TH_H - 1)
                # sample rows just outside bar
                above_row = y0 - 1
                below_row = y1 + 1
                # Convert to padded coords
                xpad0 = max(0, x1) + 1
                xpad1 = min(W - 1, x2) + 1
                is_peri = False
                if 0 <= above_row < H:
                    if np.any(outside[above_row + 1, xpad0:xpad1+1] == 1):
                        is_peri = True
                if 0 <= below_row < H and not is_peri:
                    if np.any(outside[below_row + 1, xpad0:xpad1+1] == 1):
                        is_peri = True
                if is_peri:
                    edges.append(('h', y, x1, x2))

        # Check each vertical centerline for adjacency to outside
        if v_arr is not None and len(v_arr) > 0:
            for (x, y1), (_, y2) in v_arr.tolist():
                x = int(x)
                y1 = int(min(y1, y2)); y2 = int(max(y1, y2))
                x0 = max(0, x - TH_V // 2)
                x1 = min(W - 1, x0 + TH_V - 1)
                left_col = x0 - 1
                right_col = x1 + 1
                ypad0 = max(0, y1) + 1
                ypad1 = min(H - 1, y2) + 1
                is_peri = False
                if 0 <= left_col < W:
                    if np.any(outside[ypad0:ypad1+1, left_col + 1] == 1):
                        is_peri = True
                if 0 <= right_col < W and not is_peri:
                    if np.any(outside[ypad0:ypad1+1, right_col + 1] == 1):
                        is_peri = True
                if is_peri:
                    edges.append(('v', x, y1, y2))

        # Deduplicate
        edges = list({e: None for e in edges}.keys())
        return edges

    @staticmethod
    def _compute_perimeter_edges(rect_bounds: np.ndarray):
        """Return outer perimeter edges of the union of rectangles as canonical tuples.
        Edges are ('h', y, x1, x2) or ('v', x, y1, y2) with half-open intervals on x2/y2.
        Only edges that border the true outside (via flood-fill) are returned.
        """
        if rect_bounds is None or len(rect_bounds) == 0:
            return []

        # Determine canvas extents
        # We assume coordinates lie within [0..WIDTH) x [0..HEIGHT)
        # Build occupancy grid (inside union of rectangles): half-open fills [y1:y2, x1:x2]
        H = int(HEIGHT)
        W = int(WIDTH)
        inside = np.zeros((H, W), dtype=np.uint8)
        for x1, y1, x2, y2 in rect_bounds.tolist():
            x1 = max(0, min(W, int(x1)))
            x2 = max(0, min(W, int(x2)))
            y1 = max(0, min(H, int(y1)))
            y2 = max(0, min(H, int(y2)))
            if x2 > x1 and y2 > y1:
                inside[y1:y2, x1:x2] = 1

        # Flood fill outside on a padded grid to detect connectivity to the exterior
        H2, W2 = H + 2, W + 2
        pad = np.zeros((H2, W2), dtype=np.uint8)
        pad[1:H+1, 1:W+1] = inside
        # outside mask where 0s are background; we'll mark reachable 0s starting at (0,0)
        from collections import deque
        outside = np.zeros_like(pad, dtype=np.uint8)
        q = deque()
        q.append((0, 0))
        outside[0, 0] = 1
        while q:
            y, x = q.popleft()
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < H2 and 0 <= nx < W2 and outside[ny, nx] == 0 and pad[ny, nx] == 0:
                    outside[ny, nx] = 1
                    q.append((ny, nx))

        # Helper to test if the side beyond an edge reaches the true outside
        def side_is_outside_h(y, x1, x2, above=True):
            # sample just outside: for padded outside, inside[y, x] maps to outside[y+1, x+1]
            # Edge at y; above -> sample row y-1; below -> sample row y
            yr = (y - 1) if above else y
            if yr < 0 or yr >= H:
                return True  # beyond bounds is outside
            ypad = yr + 1
            xpad_start = max(0, x1) + 1
            xpad_end = min(W, x2) + 1
            return np.any(outside[ypad, xpad_start:xpad_end] == 1)

        def side_is_outside_v(x, y1, y2, left=True):
            xr = (x - 1) if left else x
            if xr < 0 or xr >= W:
                return True
            xpad = xr + 1
            ypad_start = max(0, y1) + 1
            ypad_end = min(H, y2) + 1
            return np.any(outside[ypad_start:ypad_end, xpad] == 1)

        # Gather candidate edges from rectangle bounds, then keep sides that face the true outside
        edges = []
        for x1, y1, x2, y2 in rect_bounds.tolist():
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
            if x2 > x1 and y2 > y1:
                # top (outside above)
                if side_is_outside_h(y1, x1, x2, above=True):
                    edges.append(('h', y1, x1, x2))
                # bottom (outside below)
                if side_is_outside_h(y2, x1, x2, above=False):
                    edges.append(('h', y2, x1, x2))
                # left (outside left)
                if side_is_outside_v(x1, y1, y2, left=True):
                    edges.append(('v', x1, y1, y2))
                # right (outside right)
                if side_is_outside_v(x2, y1, y2, left=False):
                    edges.append(('v', x2, y1, y2))

        # Deduplicate edges
        edges = list({e: None for e in edges}.keys())
        return edges

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
- Corner:- Corner mode: Left-click toggles a corner on/off. Corners are snapped by settings below.
- Connect: Left-click a start corner, then an aligned end corner to toggle segments between them. Right-click cancels selection.
- Player/Enemy: Author starting positions for Player and Enemies directly on path segments.

Placement (Player & Enemies)
- Switch to the correct mode.
- Click on a path to place:
    • Player: Only one position is stored; a new click moves it.
    • Enemy: Each click appends an enemy position.
- Right-click removes the nearest player/enemy within a small radius.
- Clear enemies: Remove all enemy placements. Right-click near the player removes the player.
- The first enemy placed on the perimiter is chosen as the tracer automatically.

Template
- Show: Toggle background rendering of the original maze.
- Snap template: Snap to the original template's grid (from amidar_mazes.original).
- Add corners: Add all template corners to your current set.
- Add player & enemies: Loads MAX_ENEMIES, INITIAL_PLAYER_POSITION and INITIAL_ENEMY_POSITIONS from the original maze module.
    • These are defined in src/jaxatari/games/amidar_mazes/original.py.

Tools
- Snap axis: Snap to existing corners' X/Y to keep lines straight.
- Show perimeter: Toggle a cyan dashed overlay of the maze's outer perimeter for visual verification.
- Show rectangles: Toggle an orange outline with an X and index over each detected rectangle for visual verification.
- Connect all: Connect all adjacent corners along rows and columns where valid.
- Clear: Remove all corners, edges, enemies and the player.

Export or Ctrl+S: Writes a Python module with JAX arrays into src/jaxatari/games/amidar_mazes/maze_N.py, including:
    • PATH_CORNERS, HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES, PATH_EDGES
    • RECTANGLES, RECTANGLE_BOUNDS, CORNER_RECTANGLES
    • MAX_ENEMIES, INITIAL_PLAYER_POSITION, INITIAL_ENEMY_POSITIONS
    • PLAYER_STARTING_PATH

Validation
- All necessary conditions for the for a maze are validated automatically.
- If validation fails, export is blocked with a helpful message.

Corner rectangles (manual selection fallback)
- The editor auto-detects four corner rectangles for chicken mode.
- If it can't find 4 distinct corner rectangles, you'll be prompted to click-select them:
    • Rectangle bounds will be shown; click a rectangle to toggle selection (green = selected).
    • Select exactly four, then click Export again.

Tips
- Use the background Template and snapping to stay aligned with the original design.
- The status bar shows coordinates and snapping hints under the cursor.
"""
        txt.insert("1.0", guide)
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
        self._invalidate_rect_cache(graph_changed=True)
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
        self._invalidate_rect_cache(graph_changed=True)
        self._redraw()

    def _add_template_corners(self):
        if not TEMPLATE_CORNERS:
            messagebox.showwarning("No template", "No template corners available.")
            return
        before = len(self.corners)
        self.corners.update(TEMPLATE_CORNERS)
        added = len(self.corners) - before
        if added:
            self._invalidate_rect_cache(graph_changed=True)
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
        changed = False
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
                    changed = True
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
                    changed = True
        if changed:
            self._invalidate_rect_cache(graph_changed=True)

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
        # Topology may have changed; ensure caches are fresh
        self._invalidate_rect_cache(graph_changed=True)
        self._redraw()
        self.status.config(text=f"Connected all possible corners.")

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
            self._invalidate_rect_cache(graph_changed=True)
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
            self._invalidate_rect_cache(graph_changed=True)

        self._redraw()

    # --------------------- Events ---------------------
    def _on_left_click(self, ev):
        x, y = self._screen_to_world(ev.x, ev.y)
        # Apply snapping for easier aligned placement
        x, y, _, _ = self._snap_point(x, y)

        # Corner-rectangle manual selection overrides
        if self.corner_select_active:
            if self._rect_bounds_cache is None or len(self._rect_bounds_cache) == 0:
                messagebox.showwarning("Corner selection", "No rectangles available to select. Connect the maze into rectangles first.")
                return
            self._toggle_corner_rectangle_selection_at(x, y)
            return
        if self.tool.get() == "corner":
            # Toggle corner on left click
            near = self._nearest_corner(x, y)
            if near is None:
                self._add_corner(x, y)
            else:
                self._remove_corner(*near)
                self.pending_connect = None
                # _remove_corner calls _redraw()
        elif self.tool.get() == "connect":
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
        else:
            # place mode
            if not self._is_on_path_point(x, y):
                messagebox.showwarning("Not on path", "Please place entities on a path segment.")
                return
            if self.tool.get() == "player":
                self.player_pos = (x, y)
            else:
                self.enemy_positions.append((x, y))
            self._redraw()
        self._on_motion(ev)

    def _on_right_click(self, ev):
        # remove corner in corner mode; in connect mode, right-click clears selection; in place mode, right-click removes nearest entity
        x, y = self._screen_to_world(ev.x, ev.y)
        x, y, _, _ = self._snap_point(x, y)
        if self.tool.get() == "corner":
            # remove corner on right click
            near = self._nearest_corner(x, y)
            if near is not None:
                self._remove_corner(*near)
                self.pending_connect = None
                # _remove_corner calls _redraw()
        elif self.tool.get() == "connect":
            self.pending_connect = None
            self._redraw()
        elif self.tool.get() == "player":
            # remove player if within radius
            if self.player_pos is not None:
                px, py = self.player_pos
                if abs(px - x) + abs(py - y) <= max(6, 2 * self.scale):
                    self.player_pos = None
                    self._redraw()
        elif self.tool.get() == "enemy":
            x, y = self._screen_to_world(ev.x, ev.y)
            x, y, _, _ = self._snap_point(x, y)
            # remove nearest enemy within radius
            if self.enemy_positions:
                idx, d = self._nearest_enemy_index(x, y)
                if idx is not None and d <= max(6, 2 * self.scale):
                    self.enemy_positions.pop(idx)
                    self._redraw()
        self._on_motion(ev)

    def _on_motion(self, ev):
        x, y = self._screen_to_world(ev.x, ev.y)
        sx, sy, use_x, use_y = self._snap_point(x, y)
        info_place = ""
        if self.tool.get() == "player":
            info_place = f" | player={self.player_pos}"
        if self.tool.get() == "enemy":
            info_place = f" | enemies={len(self.enemy_positions)}"
        self.status.config(text=f"Tool: {self.tool.get()} | x={x}→{sx if use_x else x}, y={y}→{sy if use_y else y} | corners={len(self.corners)} | horizontal edges={len(self.h_edges)} | vertical edges={len(self.v_edges)}{info_place}")
        self._draw_guides(sx, sy, use_x, use_y)

    def _on_clear(self):
        if messagebox.askyesno("Clear", "Remove everything?"):
            self.player_pos = None
            self.enemy_positions.clear()
            self.corners.clear()
            self.h_edges.clear()
            self.v_edges.clear()
            self.pending_connect = None
            self._invalidate_rect_cache(graph_changed=True)
            self._redraw()
            self.status.config(text=f"Cleared.")

    # --------------------- Draw ---------------------
    def _redraw(self):
        self.canvas.delete("all")

        # Ensure we have rectangle bounds cached for perimeter overlay if possible
        try:
            if self._rect_bounds_cache is None:
                corners_arr, h_arr, v_arr = self._assemble_arrays()
                if len(corners_arr) > 0 and (len(h_arr) > 0 or len(v_arr) > 0):
                    _, rect_bounds = self._compute_rectangles(corners_arr, h_arr, v_arr)
                    self._rect_bounds_cache = rect_bounds
        except Exception:
            # Don't block render on preview cache failures
            pass

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

        # Draw placements (player/enemies)
        if self.player_pos is not None:
            px, py = self.player_pos
            sx, sy = self._world_to_screen(px, py)
            rr = max(2, int(2 * self.scale))
            self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline="#00bfff", width=2, fill="")
            self.canvas.create_text(sx, sy - 8, text="P", fill="#00bfff", font=("Segoe UI", max(6, int(6 * self.scale / 3))))
        for idx, (ex, ey) in enumerate(self.enemy_positions):
            sx, sy = self._world_to_screen(ex, ey)
            rr = max(2, int(2 * self.scale))
            self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline="#ff66cc", width=2, fill="")
            self.canvas.create_text(sx, sy - 8, text=str(idx), fill="#ff66cc", font=("Segoe UI", max(6, int(6 * self.scale / 3))))

        # Optional: Draw detected perimeter edges (outer boundary) for visual verification
        # Prefer using current edges directly to compute perimeter, so the overlay updates live
        if self.show_perimeter.get():
            try:
                _, h_arr_prev, v_arr_prev = self._assemble_arrays()
                peri = self._compute_perimeter_edges_from_edges(h_arr_prev, v_arr_prev)
            except Exception:
                peri = []
            if peri:
                for e in peri:
                    if e[0] == 'h':
                        _, yy, x1, x2 = e
                        sx1, sy = self._world_to_screen(x1, yy)
                        sx2, _ = self._world_to_screen(x2, yy)
                        self.canvas.create_line(sx1, sy, sx2, sy, fill="#00ffff", width=max(1, int(self.scale)), dash=(3, 2))
                    else:
                        _, xx, y1, y2 = e
                        sx, sy1 = self._world_to_screen(xx, y1)
                        _, sy2 = self._world_to_screen(xx, y2)
                        self.canvas.create_line(sx, sy1, sx, sy2, fill="#00ffff", width=max(1, int(self.scale)), dash=(3, 2))

        # Draw rectangle bounds for manual corner selection
        if self.corner_select_active and self._rect_bounds_cache is not None and len(self._rect_bounds_cache) > 0:
            for i, (rx1, ry1, rx2, ry2) in enumerate(self._rect_bounds_cache.tolist()):
                sx1, sy1 = self._world_to_screen(int(rx1), int(ry1))
                sx2, sy2 = self._world_to_screen(int(rx2), int(ry2))
                sel = i in set(self.corner_rectangle_overrides)
                color = "#66ff66" if sel else "#aaaaaa"
                width = 3 if sel else 1
                self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=width, dash=() if sel else (4, 3))
            # Status hint
            self.status.config(text=f"Select 4 corner rectangles: {len(self.corner_rectangle_overrides)}/4 selected. Click rectangles to toggle. Click Export to save when 4 are selected.")

        # Optional: draw all rectangle bounds with an X and index for debugging
        if self.show_rectangles.get():
            try:
                if self._rect_bounds_cache is None:
                    corners_arr, h_arr, v_arr = self._assemble_arrays()
                    if len(corners_arr) > 0 and (len(h_arr) > 0 or len(v_arr) > 0):
                        _, rect_bounds = self._compute_rectangles(corners_arr, h_arr, v_arr)
                        self._rect_bounds_cache = rect_bounds
                bounds = self._rect_bounds_cache if self._rect_bounds_cache is not None else []
            except Exception:
                bounds = []
            for i, (rx1, ry1, rx2, ry2) in enumerate(getattr(bounds, 'tolist', lambda: [])()):
                sx1, sy1 = self._world_to_screen(int(rx1), int(ry1))
                sx2, sy2 = self._world_to_screen(int(rx2), int(ry2))
                # rectangle outline
                self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#ff8800", width=2)
                # draw an X inside
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#ff8800", width=1)
                self.canvas.create_line(sx1, sy2, sx2, sy1, fill="#ff8800", width=1)
                # label with index
                cx = (sx1 + sx2) // 2
                cy = (sy1 + sy2) // 2
                self.canvas.create_text(cx, cy, text=str(i), fill="#ffffff", font=("Segoe UI", max(6, int(6 * self.scale / 3))))

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
        # Validate placements
        if self.player_pos is None:
            return False, "Please place the initial player position."
        # Must be on path
        if not self._is_on_path_point(*self.player_pos):
            return False, f"Player position {self.player_pos} is not on a path."
        for i, pos in enumerate(self.enemy_positions):
            if not self._is_on_path_point(*pos):
                return False, f"Enemy {i} at {pos} is not on a path."
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
        # Ensure consistency: split edges at corners before all downstream computations
        h_arr, v_arr = MazeEditor._split_edges_at_corners(corners_arr, h_arr, v_arr)

        # Precompute derived path constants (rectangles, bounds)
        rect_masks, rect_bounds = self._compute_rectangles(corners_arr, h_arr, v_arr)

        # Prepare placements
        player_pos = np.array(list(self.player_pos), dtype=np.int32)

        # Robust perimeter detection based on flood-fill of the outside
        def _point_on_edges(pt, edges):
            x, y = pt
            if not self._is_on_path_point(x, y):
                return False
            for e in edges:
                if e[0] == 'h':
                    _, yy, x1, x2 = e
                    if y == yy and x1 <= x <= x2:
                        return True
                else:
                    _, xx, y1, y2 = e
                    if x == xx and y1 <= y <= y2:
                        return True
            return False

        perimeter_edges = self._compute_perimeter_edges_from_edges(h_arr, v_arr)

        # Work on a copy to possibly reorder
        enemy_positions = list(self.enemy_positions)
        max_enemies = len(enemy_positions)
        perimeter_indices = [i for i, p in enumerate(enemy_positions) if _point_on_edges(p, perimeter_edges)]
        if not perimeter_indices:
            messagebox.showerror(
                "Tracer placement",
                "Place at least one enemy on the outer perimeter to act as the Tracer, then export again.",
            )
            return
        # Move the first perimeter enemy to index 0 (stable order for others)
        first_idx = perimeter_indices[0]
        if first_idx != 0:
            tracer = enemy_positions.pop(first_idx)
            enemy_positions.insert(0, tracer)
            # reflect change in UI state so user sees the order
            self.enemy_positions = enemy_positions[:]

        enemy_pos = np.array(enemy_positions, dtype=np.int32)

        # Compute player's starting path index (index into PATH_EDGES = [H|V])
        def _compute_player_starting_path(h_edges: np.ndarray, v_edges: np.ndarray, player: np.ndarray) -> int:
            px, py = int(player[0]), int(player[1])
            # Check horizontals first (order matters; matches runtime concat order)
            for i in range(h_edges.shape[0]):
                x1, y = int(h_edges[i, 0, 0]), int(h_edges[i, 0, 1])
                x2 = int(h_edges[i, 1, 0])
                if py == y and x1 <= px <= x2:
                    return i
            base = h_edges.shape[0]
            for j in range(v_edges.shape[0]):
                x, y1 = int(v_edges[j, 0, 0]), int(v_edges[j, 0, 1])
                y2 = int(v_edges[j, 1, 1])
                if px == x and y1 <= py <= y2:
                    return base + j
            # Fallback (shouldn't happen due to earlier validation)
            return 0

        player_start_idx = _compute_player_starting_path(h_arr, v_arr, player_pos)

        # Manual-corner selection flow
        if self.corner_select_active:
            # Ensure cache is populated for drawing
            self._rect_bounds_cache = rect_bounds
            self._rect_masks_cache = rect_masks
            if len(self.corner_rectangle_overrides) != 4:
                self._redraw()
                messagebox.showerror("Corner rectangles", "Please select exactly 4 corner rectangles before exporting.")
                return
            corner_rect_idxs = np.array(self.corner_rectangle_overrides, dtype=np.int32)
        else:
            # Try auto-detection first
            corner_rect_idxs = self._compute_corner_rectangles(rect_bounds)
            unique_idxs = sorted(set(int(i) for i in corner_rect_idxs.tolist())) if rect_bounds.size else []
            # Not 4 distinct -> require selection
            if len(unique_idxs) < 4:
                self._rect_bounds_cache = rect_bounds
                self._rect_masks_cache = rect_masks
                self.corner_rectangle_overrides = []
                self.corner_select_active = True
                self._redraw()
                messagebox.showwarning(
                    "Corner rectangles",
                    "Could not auto-detect 4 distinct corner rectangles.\n\nPlease click the four corner rectangles in the canvas to select them, then click Export again.",
                )
                return

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
        content.append(f"MAX_ENEMIES = {max_enemies}")
        content.append("")
        content.append(f"PATH_CORNERS = jnp.array({corners_arr.tolist()}, dtype=jnp.int32)")
        content.append(f"HORIZONTAL_PATH_EDGES = jnp.array({h_arr.tolist()}, dtype=jnp.int32)")
        content.append(f"VERTICAL_PATH_EDGES = jnp.array({v_arr.tolist()}, dtype=jnp.int32)")
        # Provide PATH_EDGES and precomputed rectangle data to avoid recomputation in consumers
        content.append("PATH_EDGES = jnp.concatenate((HORIZONTAL_PATH_EDGES, VERTICAL_PATH_EDGES), axis=0)")
        content.append(f"RECTANGLES = jnp.array({rect_masks.tolist()}, dtype=jnp.int32)")
        content.append(f"RECTANGLE_BOUNDS = jnp.array({rect_bounds.tolist()}, dtype=jnp.int32)")
        content.append(f"CORNER_RECTANGLES = jnp.array({corner_rect_idxs.tolist()}, dtype=jnp.int32)")
        content.append("")
        content.append(f"INITIAL_PLAYER_POSITION = jnp.array({player_pos.tolist()}, dtype=jnp.int32)")
        content.append(f"INITIAL_ENEMY_POSITIONS = jnp.array({enemy_pos.tolist()}, dtype=jnp.int32)")
        content.append(f"PLAYER_STARTING_PATH = jnp.array([{int(player_start_idx)}], dtype=jnp.int32)")
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

        # Clear selection state after a successful save
        if self.corner_select_active:
            self.corner_select_active = False
            self.corner_rectangle_overrides = []
            self._rect_bounds_cache = None
            self._rect_masks_cache = None

    # --------------------- Placement helpers ---------------------
    def _is_on_path_point(self, x: int, y: int) -> bool:
        """Return True if (x,y) lies on any horizontal or vertical edge segment."""
        # Check horizontals
        for (a, b) in self.h_edges:
            (x1, yy), (x2, _) = a, b
            if y == yy and min(x1, x2) <= x <= max(x1, x2):
                return True
        # Check verticals
        for (a, b) in self.v_edges:
            (xx, y1), (_, y2) = a, b
            if x == xx and min(y1, y2) <= y <= max(y1, y2):
                return True
        return False

    def _nearest_enemy_index(self, x: int, y: int):
        """Return (index, L1 distance) of the nearest enemy; (None, inf) if none."""
        if not self.enemy_positions:
            return None, float("inf")
        dists = [abs(ex - x) + abs(ey - y) for (ex, ey) in self.enemy_positions]
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    def _toggle_corner_rectangle_selection_at(self, x: int, y: int):
        """Toggle selection of a rectangle whose bounds contain (x,y)."""
        if self._rect_bounds_cache is None or len(self._rect_bounds_cache) == 0:
            return
        # Find top-most matching rect (smallest area first heuristic)
        bounds = self._rect_bounds_cache
        areas = (bounds[:, 2] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 1])
        order = np.argsort(areas)
        hit = None
        for i in order.tolist():
            rx1, ry1, rx2, ry2 = map(int, bounds[i].tolist())
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                hit = i
                break
        if hit is None:
            return
        # Toggle membership; cap at 4 selections
        if hit in self.corner_rectangle_overrides:
            self.corner_rectangle_overrides.remove(hit)
        else:
            if len(self.corner_rectangle_overrides) >= 4:
                messagebox.showinfo("Corner rectangles", "You already selected 4 rectangles. Click one again to deselect.")
                return
            self.corner_rectangle_overrides.append(hit)
        self._redraw()


    def _use_original_positions(self):
        """Populate player and enemies from the original maze module if available."""
        try:
            max_e = int(getattr(ORIG, "MAX_ENEMIES", 6))
            ply = getattr(ORIG, "INITIAL_PLAYER_POSITION", None)
            enm = getattr(ORIG, "INITIAL_ENEMY_POSITIONS", None)

            # Fallbacks if missing
            if ply is None:
                ply = np.array([140, 89], dtype=np.int32)
            if enm is None:
                enm = np.array([
                    [16, 14],
                    [16, 14],
                    [44, 14],
                    [16, 137],
                    [52, 164],
                    [16, 164],
                ], dtype=np.int32)

            # Set UI state
            self.player_pos = (int(ply[0]), int(ply[1]))
            self.enemy_positions = [(int(x), int(y)) for x, y in np.asarray(enm).tolist()][:max_e]
            self._redraw()
            self.status.config(text=f"Added player and {max_e} enemies at original positions.")
        except Exception as e:
            messagebox.showerror("Load originals", f"Failed to load original positions: {e}")
        self._redraw()


if __name__ == "__main__":
    app = MazeEditor()
    app.mainloop()