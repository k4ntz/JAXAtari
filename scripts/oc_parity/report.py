"""Visual + JSON reports for lockstep OC→JAX evaluation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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

    try:
        from PIL import Image

        Image.fromarray(strip).save(path)
    except ImportError:
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, strip)
        except ImportError:
            np.save(path + ".npy", strip)


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
    """Save OC|JAX|diff strips for selected relative frames in a run."""
    frames = run.get("strip_frames")
    if not frames:
        return
    sub = os.path.join(out_dir, "strips", label)
    _ensure_dir(sub)
    for k in sample_ks:
        if k < 0 or k >= len(frames):
            continue
        pair = frames[k]
        if pair is None:
            continue
        oc_f, jax_f = pair
        save_side_by_side_strip(os.path.join(sub, f"k{k:03d}.png"), oc_f, jax_f)
