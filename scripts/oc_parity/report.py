"""Visual + JSON reports for lockstep OC→JAX evaluation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_rgb_image(path: str, strip: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(strip).save(path)
    except ImportError:
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, strip)
        except ImportError:
            np.save(path + ".npy", strip)


def save_side_by_side_strip(
    path: str,
    oc_frame: np.ndarray,
    jax_frame: np.ndarray,
    *,
    scale: int = 2,
) -> None:
    """Write OC | JAX | |diff| PNG (or .npy fallback)."""
    oc = np.asarray(oc_frame, dtype=np.uint8)
    jf = np.asarray(jax_frame, dtype=np.uint8)
    h = min(oc.shape[0], jf.shape[0])
    w = min(oc.shape[1], jf.shape[1])
    oc = oc[:h, :w]
    jf = jf[:h, :w]
    diff = np.clip(np.abs(oc.astype(np.int16) - jf.astype(np.int16)), 0, 255).astype(
        np.uint8
    )
    strip = np.concatenate([oc, jf, diff], axis=1)
    if scale > 1:
        strip = np.repeat(np.repeat(strip, scale, axis=0), scale, axis=1)
    _save_rgb_image(path, strip)


def save_change_map_strip(
    path: str,
    d_oc: np.ndarray,
    d_jax: np.ndarray,
    d_diff: np.ndarray,
    *,
    scale: int = 2,
) -> None:
    """Write |OC−oc0| | |JAX−jax0| | ||Δoc|−|Δjax|| PNG."""
    a = np.asarray(d_oc, dtype=np.uint8)
    b = np.asarray(d_jax, dtype=np.uint8)
    c = np.asarray(d_diff, dtype=np.uint8)
    h = min(a.shape[0], b.shape[0], c.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1])
    strip = np.concatenate([a[:h, :w], b[:h, :w], c[:h, :w]], axis=1)
    if scale > 1:
        strip = np.repeat(np.repeat(strip, scale, axis=0), scale, axis=1)
    _save_rgb_image(path, strip)


def _motion_magnitude(frame_t: np.ndarray, frame_0: np.ndarray) -> np.ndarray:
    """Per-pixel mean |ΔRGB| as float32."""
    a = np.asarray(frame_t, dtype=np.float32)
    b = np.asarray(frame_0, dtype=np.float32)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    if a.ndim == 3:
        return np.abs(a - b).mean(axis=2)
    return np.abs(a - b)


def _apply_colormap(mag: np.ndarray, *, vmax: Optional[float] = None) -> np.ndarray:
    """Map scalar field → RGB uint8 with a magma-like colormap."""
    m = np.asarray(mag, dtype=np.float32)
    lo = float(np.nanmin(m)) if m.size else 0.0
    hi = float(vmax) if vmax is not None else float(np.nanmax(m)) if m.size else 1.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm

        rgba = cm.magma(norm)  # HxWx4 float
        return (rgba[..., :3] * 255.0).astype(np.uint8)
    except ImportError:
        # Grayscale fallback
        g = (norm * 255.0).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


def save_motion_heatmap_strip(
    path: str,
    oc0: np.ndarray,
    jax0: np.ndarray,
    oc_t: np.ndarray,
    jax_t: np.ndarray,
    *,
    scale: int = 2,
    title_note: str = "",
) -> None:
    """Write OC-motion | JAX-motion | mismatch heatmaps (t0→t).

    Motion = mean |RGB| change from inject frame; mismatch = |H_oc − H_jax|.
    """
    h_oc = _motion_magnitude(oc_t, oc0)
    h_jax = _motion_magnitude(jax_t, jax0)
    h = min(h_oc.shape[0], h_jax.shape[0])
    w = min(h_oc.shape[1], h_jax.shape[1])
    h_oc, h_jax = h_oc[:h, :w], h_jax[:h, :w]
    h_mis = np.abs(h_oc - h_jax)
    vmax = float(max(np.nanmax(h_oc), np.nanmax(h_jax), 1.0))
    rgb_oc = _apply_colormap(h_oc, vmax=vmax)
    rgb_jax = _apply_colormap(h_jax, vmax=vmax)
    rgb_mis = _apply_colormap(h_mis, vmax=vmax)
    strip = np.concatenate([rgb_oc, rgb_jax, rgb_mis], axis=1)
    if scale > 1:
        strip = np.repeat(np.repeat(strip, scale, axis=0), scale, axis=1)
    _save_rgb_image(path, strip)
    del title_note  # reserved for future labeled panels



def write_summary_json(path: str, summary: Mapping[str, Any]) -> None:
    def _conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_conv(dict(summary)), f, indent=2)


def plot_time_series(
    path: str,
    curves: Mapping[str, Sequence[float]],
    *,
    title: str,
    xlabel: str = "frame k",
    vline: Optional[float] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: save raw npz of curves
        np.savez_compressed(path + ".npz", **{k: np.asarray(v) for k, v in curves.items()})
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, ys in curves.items():
        ax.plot(np.arange(len(ys)), ys, label=name)
    if vline is not None and np.isfinite(vline):
        ax.axvline(vline, color="red", linestyle="--", label=f"median diverge@{vline:.0f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_diverge_histogram(
    path: str,
    first_diverges: Sequence[Optional[int]],
    n: int,
    *,
    title: str = "First-diverge distribution",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        vals = [n if fd is None else int(fd) for fd in first_diverges]
        np.save(path + ".npy", np.asarray(vals))
        return

    vals = [n if fd is None else int(fd) for fd in first_diverges]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=min(40, max(n // 5, 5)), range=(0, n), color="steelblue", edgecolor="white")
    ax.set_xlabel("first diverge frame (n = survived)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def select_representative_runs(
    runs: Sequence[Mapping[str, Any]],
    n: int,
    *,
    key_field: str = "first_diverge_soft",
) -> Dict[str, Mapping[str, Any]]:
    """Pick best / median / worst by a first-diverge field (None = best)."""
    if not runs:
        return {}

    def key(r):
        fd = r.get(key_field, r.get("first_diverge"))
        return n + 1 if fd is None else int(fd)

    ordered = sorted(runs, key=key, reverse=True)
    best = ordered[0]
    worst = ordered[-1]
    mid = ordered[len(ordered) // 2]
    return {"best": best, "median": mid, "worst": worst}


def write_run_strips(
    out_dir: str,
    label: str,
    run: Mapping[str, Any],
    sample_ks: Sequence[int],
) -> None:
    """Save OC|JAX|diff, change-map, and t0→t* motion heatmap strips."""
    frames = run.get("strip_frames")
    if not frames:
        return
    sub = os.path.join(out_dir, "strips", label)
    _ensure_dir(sub)
    change_frames = run.get("change_strip_frames")
    for k in sample_ks:
        if k < 0 or k >= len(frames):
            continue
        pair = frames[k]
        if pair is None:
            continue
        oc_f, jax_f = pair
        save_side_by_side_strip(os.path.join(sub, f"k{k:03d}.png"), oc_f, jax_f)
        if change_frames is not None and k < len(change_frames) and change_frames[k] is not None:
            d_oc, d_jax, d_diff = change_frames[k]
            save_change_map_strip(
                os.path.join(sub, f"k{k:03d}_change.png"), d_oc, d_jax, d_diff
            )

    # Motion heatmap: inject → soft-diverge (or last frame).
    motion = run.get("motion_pairs")
    if not motion or motion[0] is None:
        return
    n = int(run.get("n", len(motion)))
    fd = run.get("first_diverge_soft")
    k_end = (n - 1) if fd is None else int(min(max(int(fd), 0), n - 1))
    if motion[k_end] is None:
        # Fall back to last available sample in strip_ks.
        for k in reversed(list(sample_ks)):
            if 0 <= k < len(motion) and motion[k] is not None:
                k_end = k
                break
    if motion[k_end] is None:
        return
    oc0, jax0 = motion[0]
    oc_t, jax_t = motion[k_end]
    save_motion_heatmap_strip(
        os.path.join(sub, f"motion_t0_to_k{k_end:03d}.png"),
        oc0,
        jax0,
        oc_t,
        jax_t,
    )
    # Absolute diff at endpoint for problem galleries.
    save_side_by_side_strip(os.path.join(sub, f"k{k_end:03d}_endpoint.png"), oc_t, jax_t)
