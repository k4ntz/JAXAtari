#!/usr/bin/env python3
"""Paper-style start vs end lockstep gallery (OC | JAX | |diff|).

For each game, takes best (and worst) soft-survival runs from summary.json,
replays to horizon ``n``, and writes:
  - per-game PNG: start strip above end strip
  - overview collage of all best-run panels
  - index.md with survival notes

Example:
  PYTHONPATH=scripts python scripts/oc_parity/render_start_end_gallery.py \\
    --corpus data/oc_parity_corpus --cpu --n 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _crop_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _strip(oc: np.ndarray, jax_f: np.ndarray) -> np.ndarray:
    oc, jax_f = _crop_pair(oc, jax_f)
    diff = np.clip(np.abs(oc.astype(np.int16) - jax_f.astype(np.int16)), 0, 255).astype(
        np.uint8
    )
    return np.concatenate([oc, jax_f, diff], axis=1)


def _label(img: np.ndarray, text: str, *, scale: int = 2) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    bar_h = max(16, 11 * scale)
    h, w = img.shape[:2]
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:] = img
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, w, bar_h], fill=(18, 18, 18))
    draw.text((4, 2), text, fill=(230, 230, 230), font=ImageFont.load_default())
    return np.asarray(pil, dtype=np.uint8)


def _pick_best_worst(summary: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict]]:
    runs = list(summary.get("runs") or [])
    if not runs:
        return None, None
    n = int(summary.get("n", 150))

    def key(r):
        fd = r.get("first_diverge_soft")
        return n + 1 if fd is None else int(fd)

    ordered = sorted(runs, key=key, reverse=True)
    best, worst = ordered[0], ordered[-1]
    return best, worst


def _resolve_traj(corpus: str, traj_path: Optional[str]) -> Optional[str]:
    if traj_path and os.path.isfile(traj_path):
        return traj_path
    if traj_path:
        alt = os.path.join(corpus, "raw", os.path.basename(traj_path))
        if os.path.isfile(alt):
            return alt
    return None


def _replay_start_end(
    env,
    jitted_step,
    jitted_render,
    traj: Dict[str, Any],
    *,
    jax_key: str,
    t0: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (start_strip, end_strip, k_end)."""
    from oc_parity.lockstep_eval import _inject_state, _map_oc_action_to_jax

    meanings = traj.get("meta", {}).get("action_meanings")
    state = _inject_state(env, jax_key, traj, t0)
    T = len(traj["actions"])
    n_eff = min(n, max(0, T - t0))
    if n_eff <= 0:
        raise RuntimeError(f"no frames after t0={t0}")

    jax0 = np.asarray(jitted_render(state), dtype=np.uint8)
    oc0 = (
        np.asarray(traj["frames"][t0], dtype=np.uint8)
        if traj.get("frames") is not None
        else np.zeros_like(jax0)
    )
    start = _strip(oc0, jax0)

    for k in range(n_eff - 1):
        t = t0 + k
        oc_a = int(traj["actions"][t])
        jax_a = _map_oc_action_to_jax(env, oc_a, meanings)
        _o, state, _r, _d, _i = jitted_step(state, jax_a)

    k_end = n_eff - 1
    t_end = t0 + k_end
    jax_e = np.asarray(jitted_render(state), dtype=np.uint8)
    oc_e = (
        np.asarray(traj["frames"][t_end], dtype=np.uint8)
        if traj.get("frames") is not None
        else np.zeros_like(jax_e)
    )
    end = _strip(oc_e, jax_e)
    return start, end, k_end


def _stack_vertical(panels: Sequence[np.ndarray], gap: int = 4) -> np.ndarray:
    widths = [p.shape[1] for p in panels]
    w = max(widths)
    chunks = []
    for i, p in enumerate(panels):
        if p.shape[1] < w:
            pad = np.zeros((p.shape[0], w - p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        chunks.append(p)
        if i < len(panels) - 1 and gap:
            chunks.append(np.full((gap, w, 3), 40, dtype=np.uint8))
    return np.concatenate(chunks, axis=0)


def _save_png(path: str, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def _montage_grid(panels: List[Tuple[str, np.ndarray]], cols: int = 3) -> np.ndarray:
    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    # normalize widths within each row
    rows: List[np.ndarray] = []
    for i in range(0, len(panels), cols):
        batch = panels[i : i + cols]
        # pad batch to cols with black
        max_h = max(p.shape[0] for _, p in batch)
        max_w = max(p.shape[1] for _, p in batch)
        cells = []
        for name, p in batch:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[: p.shape[0], : p.shape[1]] = p
            cells.append(canvas)
        while len(cells) < cols:
            cells.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
        gap = np.full((max_h, 6, 3), 30, dtype=np.uint8)
        row = cells[0]
        for c in cells[1:]:
            row = np.concatenate([row, gap, c], axis=1)
        rows.append(row)
    hgap = 8
    out = rows[0]
    for r in rows[1:]:
        # match widths
        w = max(out.shape[1], r.shape[1])
        if out.shape[1] < w:
            out = np.concatenate(
                [out, np.zeros((out.shape[0], w - out.shape[1], 3), dtype=np.uint8)],
                axis=1,
            )
        if r.shape[1] < w:
            r = np.concatenate(
                [r, np.zeros((r.shape[0], w - r.shape[1], 3), dtype=np.uint8)],
                axis=1,
            )
        sep = np.full((hgap, w, 3), 30, dtype=np.uint8)
        out = np.concatenate([out, sep, r], axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Start/end OC|JAX|diff gallery")
    parser.add_argument("--corpus", type=str, default="data/oc_parity_corpus")
    parser.add_argument("--reports-subdir", type=str, default="reports")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir (default: corpus/final_report/start_end_gallery)",
    )
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--games", nargs="*", default=None)
    args = parser.parse_args()

    import jax
    import jaxatari
    from oc_parity.trajectory_io import load_trajectory
    from oc_parity.translators.registry import list_implemented_translators

    corpus = os.path.abspath(args.corpus)
    reports = os.path.join(corpus, args.reports_subdir)
    out_dir = args.out or os.path.join(corpus, "final_report", "start_end_gallery")
    best_dir = os.path.join(out_dir, "best")
    worst_dir = os.path.join(out_dir, "worst")
    _ensure_dir(best_dir)
    _ensure_dir(worst_dir)

    games = args.games or list_implemented_translators()
    index_lines = [
        "# Start vs end lockstep gallery",
        "",
        f"Horizon n={args.n} (~5s at 30 FPS play / 2.5s at 60 Hz ALE).",
        "Each panel: **OC | JAX | |diff|** at inject (start) and after replay (end).",
        "",
        "| game | soft_div best | soft_div worst | note |",
        "|------|---------------|----------------|------|",
    ]

    best_panels: List[Tuple[str, np.ndarray]] = []
    worst_panels: List[Tuple[str, np.ndarray]] = []

    print(f"Start/end gallery: {len(games)} games n={args.n} -> {out_dir}")

    for jax_key in games:
        summary = _load_json(os.path.join(reports, jax_key, "summary.json"))
        if summary is None:
            print(f"[skip] {jax_key}: no summary")
            continue
        best, worst = _pick_best_worst(summary)
        if best is None:
            continue

        env = jaxatari.make(jax_key)
        jitted_step = jax.jit(env.step)
        jitted_render = jax.jit(env.render)
        _o, st = env.reset(jax.random.PRNGKey(0))
        _ = jitted_render(st)

        for tag, run, dest, panel_list in (
            ("best", best, best_dir, best_panels),
            ("worst", worst, worst_dir, worst_panels),
        ):
            if run is None:
                continue
            traj_path = _resolve_traj(corpus, run.get("traj_path"))
            if traj_path is None:
                print(f"  [skip] {jax_key} {tag}: missing traj")
                continue
            t0 = int(run["t0"])
            fd = run.get("first_diverge_soft")
            fd_s = "surv" if fd is None else str(fd)
            traj = load_trajectory(traj_path)
            print(f"  {jax_key} {tag}: t0={t0} soft_div={fd_s}")
            start, end, k_end = _replay_start_end(
                env,
                jitted_step,
                jitted_render,
                traj,
                jax_key=jax_key,
                t0=t0,
                n=args.n,
            )
            if args.scale > 1:
                start = np.repeat(np.repeat(start, args.scale, axis=0), args.scale, axis=1)
                end = np.repeat(np.repeat(end, args.scale, axis=0), args.scale, axis=1)
            start = _label(
                start,
                f"{jax_key} {tag} START t0={t0}  (OC|JAX|diff)",
                scale=args.scale,
            )
            end = _label(
                end,
                f"{jax_key} {tag} END k={k_end} soft_div={fd_s}  (OC|JAX|diff)",
                scale=args.scale,
            )
            panel = _stack_vertical([start, end], gap=6)
            out_path = os.path.join(dest, f"{jax_key}.png")
            _save_png(out_path, panel)
            panel_list.append((jax_key, panel))
            if tag == "best":
                note = (
                    "hard to tell"
                    if (fd is None or int(fd) >= args.n * 0.8)
                    else (
                        "partial"
                        if (fd is not None and int(fd) >= 40)
                        else "easy to tell / early diverge"
                    )
                )
                wfd = worst.get("first_diverge_soft") if worst else None
                wfd_s = "surv" if wfd is None else str(wfd)
                index_lines.append(
                    f"| {jax_key} | {fd_s} | {wfd_s} | {note} |"
                )

    if best_panels:
        overview = _montage_grid(best_panels, cols=3)
        _save_png(os.path.join(out_dir, "overview_best_start_end.png"), overview)
        print(f"Wrote overview_best_start_end.png ({len(best_panels)} games)")
    if worst_panels:
        overview_w = _montage_grid(worst_panels, cols=3)
        _save_png(os.path.join(out_dir, "overview_worst_start_end.png"), overview_w)
        print(f"Wrote overview_worst_start_end.png ({len(worst_panels)} games)")

    index_lines += [
        "",
        "## How to read",
        "- Left→right in each strip: ALE/OC frame, JAX render, absolute pixel diff.",
        "- Top = inject (t=0); bottom = after n steps (or traj end).",
        "- If the claim is 'hard to tell after 5s', look at **best/** and especially games tagged surv / high soft_div.",
        "- Early bright diff at END ⇒ claim fails for that title without manual latent/RNG alignment.",
        "",
        f"Per-game: `{best_dir}` and `{worst_dir}`.",
    ]
    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
