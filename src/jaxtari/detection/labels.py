"""Ground-truth label geometry: how game-state objects become detector boxes.

These rules (driven by the quirks registry) define the training labels of the detectors and are used again
by the wrapper's calibration to map detections back to game coordinates. All functions take the object
data of a single frame and return boxes ``(x, y, w, h)`` in game pixels (top-left corner).
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SCREEN_W = 160.0
SCREEN_H = 210.0
MIN_BOX = 8  # label floor: boxes narrower/shorter than this are padded (centred), see pad_label

# shared 7-class taxonomy of the registry ("class_id")
CLASS_NAMES = {0: "player", 1: "enemy", 2: "projectile", 3: "collectible", 4: "structure", 5: "neutral",
               6: "enemy_projectile"}

SUPPORTED_TYPES = ("entity", "static_entity", "xy_pairs", "true_2d_grid", "grid", "flat_per_row")


@lru_cache(maxsize=None)
def load_registry(path=None):
    """The quirks registry (per game: object name -> class_id, detected_type and label geometry)."""
    path = Path(path) if path is not None else Path(__file__).with_name("quirks_registry.json")
    return json.loads(path.read_text())


def entity_boxes(obj, entry):
    """ObjectObservation-like objects (.x .y .width .height [.active]), scalar or arrays of objects."""
    if not (hasattr(obj, "x") and hasattr(obj, "y")):
        return []
    x_off, y_off = entry.get("x_offset") or 0, entry.get("y_offset") or 0
    w_pad, h_pad = entry.get("w_padding") or 0, entry.get("h_padding") or 0
    xs, ys = np.atleast_1d(np.asarray(obj.x)).ravel(), np.atleast_1d(np.asarray(obj.y)).ravel()
    ws, hs = np.atleast_1d(np.asarray(obj.width)).ravel(), np.atleast_1d(np.asarray(obj.height)).ravel()
    active = np.atleast_1d(np.asarray(obj.active)).ravel() if hasattr(obj, "active") else None
    boxes = []
    for i in range(len(xs)):
        if active is not None and not bool(active[i]):
            continue
        if xs[i] <= 1 or ys[i] <= 1:
            continue
        x, y = float(xs[i]) + x_off, float(ys[i]) + y_off
        w, h = float(ws[i]) + w_pad, float(hs[i]) + h_pad
        if (x == 0 and y == 0) or w == 0 or h == 0:
            continue
        boxes.append((x, y, w, h))
    return boxes


def xy_pair_boxes(pairs, entry):
    """(N, 2) arrays of top-left positions with a fixed box size."""
    x_off, y_off = entry.get("x_offset") or 0, entry.get("y_offset") or 0
    cw = entry.get("obj_w") or entry.get("cell_w") or 8
    ch = entry.get("obj_h") or entry.get("cell_h") or 8
    return [(float(p[0]) + x_off, float(p[1]) + y_off, cw, ch) for p in np.asarray(pairs)
            if float(p[0]) > 1 and float(p[1]) > 1]


def grid_geometry(shape, entry):
    """(rows, cols, cell_w, cell_h, step_x, step_y, origin_x, origin_y) of a tile grid (after transpose_grid)."""
    rows, cols = (shape[1], shape[0]) if entry.get("transpose_grid") else shape
    cw = entry.get("cell_w") or (SCREEN_W / cols)
    ch = entry.get("cell_h") or (SCREEN_H / rows)
    return (rows, cols, cw, ch, entry.get("step_x") or cw, entry.get("step_y") or ch,
            entry.get("grid_origin_x") or 0, entry.get("grid_origin_y") or 0)


def grid_on_value(entry):
    if entry.get("active_min") is not None:
        return float(entry["active_min"])
    return float(entry["active_value"]) if entry.get("active_value") is not None else 1.0


def grid_boxes(grid, entry):
    """(R, C) tile grids: one box per active cell."""
    grid = np.asarray(grid)
    rows, cols, cw, ch, sx, sy, ox, oy = grid_geometry(grid.shape, entry)
    if entry.get("transpose_grid"):
        grid = grid.T
    amin, av = entry.get("active_min"), entry.get("active_value")
    boxes = []
    for r in range(rows):
        for c in range(cols):
            v = float(grid[r, c])
            on = v >= float(amin) if amin is not None else v == float(av) if av is not None else v == 1.0
            if on:
                boxes.append((ox + c * sx, oy + r * sy, cw, ch))
    return boxes


def flat_per_row_boxes(row_vals, entry):
    """(H,) per-row x positions (e.g. river banks): consecutive active rows become chunks of <= max_run rows."""
    x_off, y_off = entry.get("x_offset") or 0, entry.get("y_offset") or 0
    obj_w = entry.get("obj_w") or 8
    active_v = entry.get("active_value")
    max_run = int(entry.get("max_run_height") or 32)
    row_vals = np.asarray(row_vals, dtype=float)
    active = row_vals == float(active_v) if active_v is not None else row_vals > 1
    boxes, run_start, run_xs = [], None, []

    def flush(start, xs):
        for k in range(0, len(xs), max_run):
            chunk = xs[k:k + max_run]
            boxes.append((min(chunk) + x_off, float(start + k) + y_off, (max(chunk) - min(chunk)) + float(obj_w),
                          float(len(chunk))))

    for y, on in enumerate(active):
        if on:
            if run_start is None:
                run_start, run_xs = y, []
            run_xs.append(float(row_vals[y]))
        elif run_start is not None:
            flush(run_start, run_xs)
            run_start = None
    if run_start is not None:
        flush(run_start, run_xs)
    return boxes


def label_boxes(obj, entry):
    """Unpadded label boxes of one registry object for one frame."""
    t = entry.get("detected_type")
    if t == "entity":
        return entity_boxes(obj, entry)
    if t == "xy_pairs":
        return xy_pair_boxes(obj, entry)
    if t in ("true_2d_grid", "grid"):
        return grid_boxes(obj, entry)
    if t == "flat_per_row":
        return flat_per_row_boxes(obj, entry)
    if t == "static_entity" and hasattr(obj, "x"):
        # labelled by OpenCV patches on the sprite pixels: approximated by the true object box
        return entity_boxes(obj, {})
    return []


def pad_label(entry, x, y, w, h):
    """Training-label floor: axes smaller than MIN_BOX px are padded (centred) unless cell_w / cell_h is set."""
    if entry.get("cell_w") is None and w < MIN_BOX:
        x, w = max(0, x - (MIN_BOX - w) // 2), MIN_BOX
    if entry.get("cell_h") is None and h < MIN_BOX:
        y, h = max(0, y - (MIN_BOX - h) // 2), MIN_BOX
    return x, y, w, h


def to_yolo_line(class_id, x, y, w, h, screen_w=SCREEN_W, screen_h=SCREEN_H):
    """YOLO label line (normalized centre / size, clipped to 1)."""
    return (f"{class_id} {min((x + w / 2) / screen_w, 1.0):.6f} {min((y + h / 2) / screen_h, 1.0):.6f} "
            f"{min(w / screen_w, 1.0):.6f} {min(h / screen_h, 1.0):.6f}")


def frame_labels(obs, objects_cfg, screen_w=SCREEN_W, screen_h=SCREEN_H):
    """All YOLO label lines of one observation (registry objects with an integer class_id)."""
    lines = []
    for name, entry in objects_cfg.items():
        if not isinstance(entry.get("class_id"), int):
            continue
        obj = getattr(obs, name, None)
        if obj is None:
            continue
        for box in label_boxes(obj, entry):
            x, y, w, h = pad_label(entry, *box)
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, screen_w), min(y + h, screen_h)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue  # off-screen (e.g. radar-style unclipped coordinates)
            lines.append(to_yolo_line(entry["class_id"], x0, y0, x1 - x0, y1 - y0, screen_w, screen_h))
    return lines
