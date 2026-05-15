"""
create_clean_sprites.py

- Regular sprites  → solid rectangle of the median non-transparent pixel colour,
                     same width/height as the original.
- Background arrays → hue-rotated + slightly desaturated version that preserves
                      spatial structure but is visually distinct from the originals.

Usage:
    python scripts/create_custom_sprites.py [--hue-shift DEG] [--saturation-scale S]
                                            [--shift-regular-sprites]
                                            [--regular-sprite-hue-shift DEG]
                                            [--src DIR] [--dst DIR]

Defaults:
    --src  ~/.local/share/jaxatari/sprites
    --dst  ~/.local/share/jaxatari/custom_sprites
    --hue-shift        137   (degrees, golden-angle-ish so colours stay spread out)
    --saturation-scale 0.55  (compress saturation toward grey)
    --shift-regular-sprites disabled by default
    --regular-sprite-hue-shift 18 (degrees, only used when flag is enabled)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from platformdirs import user_data_dir

DEFAULT_SRC = Path(user_data_dir("jaxatari")) / "sprites"
DEFAULT_DST = Path(user_data_dir("jaxatari")) / "custom_sprites"


# ---------------------------------------------------------------------------
# Format helpers  (matches loadFrame logic in jax_rendering_utils.py)
# ---------------------------------------------------------------------------

def opaque_mask(arr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of visible pixels for either format:
      - RGBA (H, W, 4): alpha > 0
      - RGB  (H, W, 3): not pure black (0,0,0) — same convention as loadFrame
    Works on 3D (single frame) and 4D (stacked frames) arrays.
    """
    if arr.shape[-1] == 4:
        return arr[..., 3] > 0
    else:  # 3-channel RGB, black == transparent
        return ~((arr[..., 0] == 0) & (arr[..., 1] == 0) & (arr[..., 2] == 0))


def median_color(arr: np.ndarray) -> np.ndarray:
    """Per-channel median over visible pixels; returns array matching channel count."""
    mask = opaque_mask(arr)
    visible = arr[mask]  # (N, C)
    if len(visible) == 0:
        mid = np.full(arr.shape[-1], 128, dtype=np.uint8)
        if arr.shape[-1] == 4:
            mid[3] = 255
        return mid
    result = np.median(visible, axis=0).astype(np.uint8)
    if arr.shape[-1] == 4:
        result[3] = 255  # force full alpha for the flat colour
    return result


def solid_rectangle(arr: np.ndarray, hue_shift_deg: float | None = None) -> np.ndarray:
    """
    Fill the entire bounding box with the median colour of the original visible pixels.
    The whole H×W area becomes fully opaque — no shape information is preserved.
    Handles single frames (H, W, C) and stacked batches (N, H, W, C).
    """
    if arr.ndim == 4:
        return np.stack([solid_rectangle(frame) for frame in arr])
    color = median_color(arr)
    if hue_shift_deg is not None:
        rgb = color[:3].astype(np.float32).reshape(1, 1, 3) / 255.0
        hh, ss, vv = _rgb_to_hsv_vectorised(rgb)
        hh = (hh + hue_shift_deg / 360.0) % 1.0
        shifted_rgb = (_hsv_to_rgb_vectorised(hh, ss, vv) * 255).astype(np.uint8).reshape(3)
        color = color.copy()
        color[:3] = shifted_rgb
    out = np.full_like(arr, color)
    # For RGBA, ensure alpha is fully opaque everywhere
    if arr.shape[-1] == 4:
        out[..., 3] = 255
    # For RGB, ensure no pure-black pixel (which means transparent) slips through
    # if the median happened to land on black, nudge it slightly
    else:
        is_black = (out[..., 0] == 0) & (out[..., 1] == 0) & (out[..., 2] == 0)
        if is_black.all():
            out[..., 0] = 1
    return out


def recolor_background(arr: np.ndarray, hue_shift_deg: float, sat_scale: float) -> np.ndarray:
    """
    Recolor a background while removing any text/minority colors within each region.

    Strategy: segment the image into contiguous bands sharing the same dominant color
    (using connected-component labeling on the dominant-color-per-row). Each band is
    flood-filled with a single hue-rotated version of its dominant color, making any
    minority-color text (copyright notices, labels) invisible.

    Black pixels are treated as separators and kept as-is.
    Handles single frames (H, W, C) and stacked batches (N, H, W, C).
    """
    if arr.ndim == 4:
        return np.stack([recolor_background(frame, hue_shift_deg, sat_scale) for frame in arr])

    out = arr.copy()
    h, w, c = arr.shape
    rgb = arr[..., :3]

    # 1. For each row, find the dominant (most frequent) non-black color.
    #    Rows that are purely black get dominant = None.
    row_dominant = []
    for y in range(h):
        row = rgb[y]  # (W, 3)
        non_black = row[~((row[:, 0] == 0) & (row[:, 1] == 0) & (row[:, 2] == 0))]
        if len(non_black) == 0:
            row_dominant.append(None)
        else:
            # mode: most frequent color in this row
            colors, counts = np.unique(non_black, axis=0, return_counts=True)
            row_dominant.append(tuple(colors[counts.argmax()]))

    # 2. Group consecutive rows that share the same dominant color into bands.
    bands = []  # list of (start_row, end_row_exclusive, dominant_rgb_or_None)
    start = 0
    for y in range(1, h):
        if row_dominant[y] != row_dominant[y - 1]:
            bands.append((start, y, row_dominant[y - 1]))
            start = y
    bands.append((start, h, row_dominant[h - 1]))

    # 3. For each band with a non-black dominant, hue-rotate that color and
    #    flood-fill the entire band with it.
    for (y0, y1, dom) in bands:
        if dom is None:
            continue  # pure black band — keep as-is

        dom_rgb = np.array(dom, dtype=np.float32).reshape(1, 1, 3) / 255.0
        hh, ss, vv = _rgb_to_hsv_vectorised(dom_rgb)
        hh = (hh + hue_shift_deg / 360.0) % 1.0
        ss = np.clip(ss * sat_scale, 0.0, 1.0)
        new_rgb = (_hsv_to_rgb_vectorised(hh, ss, vv) * 255).astype(np.uint8).reshape(3)

        # Fill the whole band — but keep purely black rows (separators) black
        for y in range(y0, y1):
            if row_dominant[y] is None:
                continue  # black separator row inside a band — keep
            out[y, :, :3] = new_rgb
            if c == 4:
                out[y, :, 3] = 255

    return out


def _rgb_to_hsv_vectorised(arr: np.ndarray):
    """arr shape: (H, W, 3) float32 in [0,1]."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    s = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
    v = cmax

    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h / 6.0

    return h, s, v


def _hsv_to_rgb_vectorised(h, s, v):
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_sprites(
    src: Path,
    dst: Path,
    hue_shift: float,
    sat_scale: float,
    shift_regular_sprites: bool,
    regular_sprite_hue_shift: float,
):
    npy_files = list(src.rglob("*.npy"))
    png_files = list(src.rglob("*.png"))
    print(f"Found {len(npy_files)} .npy files and {len(png_files)} .png files under {src}")

    for src_path in npy_files:
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.load(src_path)

        # Treat as background if named "background.npy", inside a "bg/" dir,
        # or if the frame is full-screen sized (largest dimension > 100px)
        frame = arr if arr.ndim == 3 else arr[0]
        is_background = (
            src_path.stem == "background"
            or "bg" in src_path.parts
            or (frame.ndim == 3 and frame.shape[0] > 100 and frame.shape[1] > 100)
        )

        if is_background:
            out_arr = recolor_background(arr, hue_shift, sat_scale)
        else:
            out_arr = solid_rectangle(
                arr,
                hue_shift_deg=regular_sprite_hue_shift if shift_regular_sprites else None,
            )

        np.save(dst_path, out_arr)

    # Copy .png files unchanged (human preview only)
    for src_path in png_files:
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    print(f"Done. Custom sprites written to {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Source sprites directory")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help="Destination custom_sprites directory")
    parser.add_argument("--hue-shift", type=float, default=137.0,
                        help="Hue rotation applied to backgrounds (degrees, default 137)")
    parser.add_argument("--saturation-scale", type=float, default=0.55,
                        help="Saturation multiplier for backgrounds (default 0.55)")
    parser.add_argument(
        "--shift-regular-sprites",
        action="store_true",
        help="If set, also hue-shift regular (non-background) sprites.",
    )
    parser.add_argument(
        "--regular-sprite-hue-shift",
        type=float,
        default=18.0,
        help="Hue rotation in degrees for regular sprites when --shift-regular-sprites is set (default 18).",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source directory not found: {args.src}\n"
                         "Run `python -m jaxatari.install_sprites` first.")

    if args.dst.exists():
        print(f"Destination {args.dst} already exists — overwriting changed files.")

    process_sprites(
        args.src,
        args.dst,
        args.hue_shift,
        args.saturation_scale,
        args.shift_regular_sprites,
        args.regular_sprite_hue_shift,
    )


if __name__ == "__main__":
    main()
