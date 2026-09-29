#!/usr/bin/env python3
"""Render side-by-side OC|JAX|diff comparison GIFs for lockstep runs.

For each game, takes two sync runs (default: best + worst soft survival from
an existing ``summary.json``) and writes one GIF where each frame shows::

    [run A: OC | JAX | |diff|]  [run B: OC | JAX | |diff|]

Example:
  PYTHONPATH=scripts python scripts/oc_parity/render_comparison_videos.py \\
    --corpus data/oc_parity_corpus --cpu --n 120 --fps 20

  # Only a few games, oracle reports:
  PYTHONPATH=scripts python scripts/oc_parity/render_comparison_videos.py \\
    --reports-subdir reports --games pong seaquest freeway --cpu
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Force JAX on CPU before importing jax when requested.
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


def _label_strip(strip: np.ndarray, text: str, scale: int) -> np.ndarray:
    """Pad a label bar above the strip (PIL)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return strip
    h, w = strip.shape[:2]
    bar_h = max(14, 10 * scale)
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:] = strip
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([0, 0, w, bar_h], fill=(20, 20, 20))
    draw.text((4, 2), text, fill=(220, 220, 220), font=font)
    return np.asarray(img, dtype=np.uint8)


def _pick_two_runs(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = list(summary.get("runs") or [])
    if not runs:
        return []
    n = int(summary.get("n", 240))

    def key(r):
        fd = r.get("first_diverge_soft")
        return n + 1 if fd is None else int(fd)

    ordered = sorted(runs, key=key, reverse=True)
    best = ordered[0]
    worst = ordered[-1]
    if best is worst or (
        best.get("t0") == worst.get("t0")
        and best.get("traj_path") == worst.get("traj_path")
    ):
        # Fall back to first two distinct t0s if possible.
        uniq = []
        seen = set()
        for r in runs:
            k = (r.get("traj_path"), r.get("t0"))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
            if len(uniq) >= 2:
                break
        return uniq
    return [best, worst]


def _replay_strips(
    env,
    jitted_step,
    jitted_render,
    traj: Dict[str, Any],
    *,
    jax_key: str,
    t0: int,
    n: int,
    oracle: bool,
    scale: int,
    label: str,
) -> List[np.ndarray]:
    from oc_parity.lockstep_eval import _inject_state, _map_oc_action_to_jax
    from oc_parity.oracle_lookahead import apply_oracle_lookahead, supports_oracle

    meanings = traj.get("meta", {}).get("action_meanings")
    state = _inject_state(env, jax_key, traj, t0)
    if oracle and supports_oracle(jax_key):
        state = apply_oracle_lookahead(
            env, state, traj, jax_key=jax_key, t0=t0, horizon=n
        )

    frames_out: List[np.ndarray] = []
    T = len(traj["actions"])
    n_eff = min(n, max(0, T - t0))
    for k in range(n_eff):
        t = t0 + k
        jax_frame = np.asarray(jitted_render(state), dtype=np.uint8)
        if traj.get("frames") is None:
            oc_frame = np.zeros_like(jax_frame)
        else:
            oc_frame = np.asarray(traj["frames"][t], dtype=np.uint8)
        strip = _strip(oc_frame, jax_frame)
        if scale > 1:
            strip = np.repeat(np.repeat(strip, scale, axis=0), scale, axis=1)
        strip = _label_strip(strip, f"{label}  t0={t0}  k={k}", scale)
        frames_out.append(strip)
        if k < n_eff - 1:
            oc_a = int(traj["actions"][t])
            jax_a = _map_oc_action_to_jax(env, oc_a, meanings)
            _o, state, _r, _d, _i = jitted_step(state, jax_a)
    return frames_out


def _stack_side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 4) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)
    out = np.zeros((h, wa + gap + wb, 3), dtype=np.uint8)
    out[:ha, :wa] = a
    out[:hb, wa + gap : wa + gap + wb] = b
    # thin separator
    out[:, wa : wa + gap] = 40
    return out


def _write_gif(path: str, frames: Sequence[np.ndarray], fps: float) -> None:
    import imageio.v2 as imageio

    duration = 1.0 / max(fps, 1e-3)
    imageio.mimsave(path, list(frames), duration=duration, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render dual lockstep comparison GIFs (best|worst) per game"
    )
    parser.add_argument("--corpus", type=str, default="data/oc_parity_corpus")
    parser.add_argument("--reports-subdir", type=str, default="reports")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir (default: corpus/final_report/comparisons)",
    )
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Frames to render per run (default: n from each game's summary.json)",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--oracle-lookahead", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests",
    )
    args = parser.parse_args()

    import jax
    import jaxatari
    from oc_parity.trajectory_io import load_trajectory
    from oc_parity.translators.registry import list_implemented_translators

    corpus = os.path.abspath(args.corpus)
    reports = os.path.join(corpus, args.reports_subdir)
    out_dir = args.out or os.path.join(corpus, "final_report", "comparisons")
    _ensure_dir(out_dir)

    games = args.games or list_implemented_translators()
    if args.max_games is not None:
        games = games[: args.max_games]

    print(
        f"Render comparisons: games={len(games)} n={args.n} fps={args.fps} "
        f"oracle={args.oracle_lookahead} -> {out_dir}"
    )

    for jax_key in games:
        summary_path = os.path.join(reports, jax_key, "summary.json")
        summary = _load_json(summary_path)
        if summary is None:
            print(f"[skip] {jax_key}: no {summary_path}")
            continue
        runs = _pick_two_runs(summary)
        if not runs:
            print(f"[skip] {jax_key}: no runs in summary")
            continue

        print(f"\n=== {jax_key}: {len(runs)} run(s) ===")
        env = jaxatari.make(jax_key)
        jitted_step = jax.jit(env.step)
        jitted_render = jax.jit(env.render)
        _o, st = env.reset(jax.random.PRNGKey(0))
        _ = jitted_render(st)

        strip_seqs: List[List[np.ndarray]] = []
        labels = []
        for i, run in enumerate(runs[:2]):
            traj_path = run.get("traj_path")
            t0 = int(run.get("t0", 0))
            fd = run.get("first_diverge_soft")
            if not traj_path or not os.path.isfile(traj_path):
                # Try corpus raw glob by basename
                alt = os.path.join(corpus, "raw", os.path.basename(str(traj_path or "")))
                if os.path.isfile(alt):
                    traj_path = alt
                else:
                    print(f"  [skip run] missing traj {traj_path}")
                    continue
            traj = load_trajectory(traj_path)
            tag = "best" if i == 0 else "worst"
            if len(runs) == 1:
                tag = "run0"
            label = f"{jax_key} {tag} soft_div={fd if fd is not None else 'surv'}"
            print(f"  replay {tag}: {os.path.basename(traj_path)} t0={t0}")
            seq = _replay_strips(
                env,
                jitted_step,
                jitted_render,
                traj,
                jax_key=jax_key,
                t0=t0,
                n=args.n,
                oracle=args.oracle_lookahead,
                scale=args.scale,
                label=label,
            )
            if seq:
                strip_seqs.append(seq)
                labels.append(tag)

        if not strip_seqs:
            print(f"  [fail] {jax_key}: no frames")
            continue

        # Align lengths; pad by repeating last frame.
        L = max(len(s) for s in strip_seqs)
        for s in strip_seqs:
            while len(s) < L:
                s.append(s[-1])

        if len(strip_seqs) == 1:
            combo = strip_seqs[0]
        else:
            combo = [
                _stack_side_by_side(strip_seqs[0][k], strip_seqs[1][k])
                for k in range(L)
            ]

        out_path = os.path.join(out_dir, f"{jax_key}_best_vs_worst.gif")
        if len(labels) == 1:
            out_path = os.path.join(out_dir, f"{jax_key}_{labels[0]}.gif")
        _write_gif(out_path, combo, args.fps)
        print(f"  wrote {out_path} ({L} frames)")

    print(f"\nDone. GIFs in {out_dir}")


if __name__ == "__main__":
    main()
