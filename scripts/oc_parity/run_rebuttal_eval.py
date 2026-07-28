#!/usr/bin/env python3
"""Default rebuttal / appendix micro-parity pipeline (clean automated transfer).

Protocol (no oracle, no manual state edits):
  1. Record random-action OCAtari trajectories (frameskip=1)
  2. Inject OC objects → JAX via automated translators at random t0
  3. Replay the same actions for ``--n`` frames (~5s)
  4. Report object L1 + pixel MAE at transfer (t=0) and at horizon end
  5. Write start|end visual strips + a short markdown table

Default games: Freeway, MontezumaRevenge, Pong, BankHeist, Seaquest, Skiing.

Prefer human-play trajs (``--traj-tag play``) when available; random remains
supported for smoke tests.

Example (after manual recordings):
  bash scripts/oc_parity/record_rebuttal_play.sh freeway
  PYTHONPATH=scripts python scripts/oc_parity/run_rebuttal_eval.py \\
    --corpus data/oc_parity_corpus --cpu --n 150 --num-t0 5 --traj-tag play
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Rebuttal showcase: strong transfer + RNG/incomplete-info controls.
DEFAULT_GAMES = (
    "freeway",
    "montezumarevenge",
    "pong",
    "bankheist",
    "seaquest",
    "skiing",
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run(cmd: List[str]) -> int:
    print(" ", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _find_trajs(raw_dir: str, jax_key: str, prefer_tag: str) -> List[str]:
    preferred = sorted(glob.glob(os.path.join(raw_dir, f"{jax_key}_{prefer_tag}*.npz")))
    if preferred:
        return preferred
    return sorted(glob.glob(os.path.join(raw_dir, f"{jax_key}_*.npz")))


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
        if x != x:
            return "nan"
        if x == float("inf") or x == float("-inf"):
            return "∞"
        return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _finite_vals(vals) -> List[float]:
    import math

    xs: List[float] = []
    for v in vals:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x and math.isfinite(x):
            xs.append(x)
    return xs


def _finite_mean(vals) -> Optional[float]:
    xs = _finite_vals(vals)
    return float(sum(xs) / len(xs)) if xs else None


def _finite_median(vals) -> Optional[float]:
    xs = sorted(_finite_vals(vals))
    if not xs:
        return None
    m = len(xs) // 2
    if len(xs) % 2:
        return float(xs[m])
    return float(0.5 * (xs[m - 1] + xs[m]))


def _per_entity_stats(runs: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-object L1 mean/median across runs for metrics_at_0 or _end field name."""
    keys: set = set()
    for r in runs:
        keys.update((r.get(field) or {}).keys())
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(keys):
        l1s = []
        missing = 0
        for r in runs:
            pe = (r.get(field) or {}).get(key) or {}
            v = pe.get("l1")
            try:
                x = float(v)
            except (TypeError, ValueError):
                missing += 1
                continue
            if x != x or not (x < float("inf")):
                missing += 1
            else:
                l1s.append(x)
        out[key] = {
            "mean_l1": _finite_mean(l1s),
            "median_l1": _finite_median(l1s),
            "n_finite": len(l1s),
            "n_missing": missing,
        }
    return out


def _pick_worst_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Worst = highest finite entity L1 at end; fallback to @0."""

    def score(r):
        import math

        for path in (
            ["metrics_at_end", "entity_mean_l1"],
            ["metrics_at_0", "entity_mean_l1"],
        ):
            try:
                x = float(r[path[0]][path[1]])
            except (KeyError, TypeError, ValueError):
                continue
            if x == x and math.isfinite(x):
                return x
        return -1.0

    if not runs:
        return None
    return max(runs, key=score)


def _render_worst_gif(
    *,
    jax_key: str,
    run: Dict[str, Any],
    n: int,
    out_path: str,
    scale: int = 2,
    fps: float = 15.0,
) -> None:
    import jax
    import jaxatari
    from oc_parity.render_comparison_videos import _replay_strips, _write_gif
    from oc_parity.trajectory_io import load_trajectory

    traj_path = run.get("traj_path")
    if not traj_path or not os.path.isfile(traj_path):
        alt = os.path.join(
            "data/oc_parity_corpus/raw", os.path.basename(str(traj_path or ""))
        )
        if os.path.isfile(alt):
            traj_path = alt
        else:
            print(f"  [skip gif] missing traj {traj_path}")
            return
    t0 = int(run["t0"])
    l1e = (run.get("metrics_at_end") or {}).get("entity_mean_l1")
    traj = load_trajectory(traj_path)
    env = jaxatari.make(jax_key)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)
    _o, st = env.reset(jax.random.PRNGKey(0))
    _ = jitted_render(st)
    label = f"{jax_key} WORST t0={t0} L1@end={_fmt(l1e)}"
    frames = _replay_strips(
        env,
        jitted_step,
        jitted_render,
        traj,
        jax_key=jax_key,
        t0=t0,
        n=n,
        oracle=False,
        scale=scale,
        label=label,
    )
    if not frames:
        print(f"  [skip gif] no frames for {jax_key}")
        return
    _ensure_dir(os.path.dirname(out_path) or ".")
    _write_gif(out_path, frames, fps)
    print(f"  wrote {out_path} ({len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuttal micro-parity eval (random trajs, auto translate, no oracle)"
    )
    parser.add_argument("--corpus", type=str, default="data/oc_parity_corpus")
    parser.add_argument(
        "--games",
        nargs="*",
        default=list(DEFAULT_GAMES),
        help=f"JAX keys (default: {' '.join(DEFAULT_GAMES)})",
    )
    parser.add_argument("--n", type=int, default=150, help="Lockstep horizon (~5s at 30 FPS)")
    parser.add_argument("--num-t0", type=int, default=5, help="Random sync points per traj")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-frames", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument(
        "--traj-tag",
        type=str,
        default="play",
        help="Trajectory filename prefix (default play → <game>_play*.npz)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record random OC trajs before eval (skips existing unless --force-record)",
    )
    parser.add_argument("--force-record", action="store_true")
    parser.add_argument(
        "--reports-subdir",
        type=str,
        default="reports_rebuttal",
        help="Where lockstep reports go (default reports_rebuttal)",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--skip-gallery",
        action="store_true",
        help="Skip start/end PNG gallery",
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Skip worst-run lockstep GIFs",
    )
    parser.add_argument("--gif-fps", type=float, default=15.0)
    parser.add_argument("--gif-scale", type=int, default=2)
    args = parser.parse_args()

    corpus = os.path.abspath(args.corpus)
    raw_dir = os.path.join(corpus, "raw")
    reports_dir = os.path.join(corpus, args.reports_subdir)
    out_dir = os.path.join(corpus, "final_report", "rebuttal")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    games = [g.lower() for g in args.games]
    py = sys.executable
    cpu = ["--cpu"] if args.cpu else []

    print(
        f"Rebuttal eval: games={games} n={args.n} num_t0={args.num_t0} "
        f"oracle=False tag={args.traj_tag}"
    )

    if args.record:
        print("\n=== Record random trajectories ===")
        rec = [
            py,
            os.path.join(_THIS_DIR, "record_corpus.py"),
            "--corpus",
            corpus,
            "--mode",
            "random",
            "--skip-frames",
            str(args.skip_frames),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--tag",
            args.traj_tag,
            "--games",
            *games,
        ]
        if args.force_record:
            rec.append("--force")
        if _run(rec) != 0:
            raise SystemExit("record_corpus failed")

    print("\n=== Lockstep (automated transfer only) ===")
    rows: List[Dict[str, Any]] = []
    for jax_key in games:
        trajs = _find_trajs(raw_dir, jax_key, args.traj_tag)
        if not trajs:
            print(f"[skip] {jax_key}: no trajs")
            continue
        game_out = os.path.join(reports_dir, jax_key)
        os.makedirs(game_out, exist_ok=True)
        # lockstep_eval --trajs takes a single path or glob string.
        traj_glob = os.path.join(raw_dir, f"{jax_key}_{args.traj_tag}*.npz")
        cmd = [
            py,
            os.path.join(_THIS_DIR, "lockstep_eval.py"),
            "-g",
            jax_key,
            "--trajs",
            traj_glob,
            "--n",
            str(args.n),
            "--t0-mode",
            "random",
            "--num-t0",
            str(args.num_t0),
            "--soft-tau",
            "12",
            "--persist",
            "5",
            "--tau",
            "2",
            "--seed",
            str(args.seed),
            "--out",
            game_out,
            *cpu,
        ]
        rc = _run(cmd)
        if rc != 0:
            print(f"[warn] {jax_key}: lockstep_eval exited {rc}")
            continue
        summary = _load_json(os.path.join(game_out, "summary.json"))
        if summary is None:
            print(f"[warn] {jax_key}: no summary")
            continue
        runs = summary.get("runs") or []
        l1_0s = [r["metrics_at_0"]["entity_mean_l1"] for r in runs]
        l1_es = [r["metrics_at_end"]["entity_mean_l1"] for r in runs]
        mae_0s = [r["metrics_at_0"]["pixel_mae"] for r in runs]
        mae_es = [r["metrics_at_end"]["pixel_mae"] for r in runs]
        per0 = _per_entity_stats(runs, "per_entity_at_0")
        pere = _per_entity_stats(runs, "per_entity_at_end")
        worst = _pick_worst_run(runs)
        rows.append(
            {
                "game": jax_key,
                "num_runs": len(runs),
                "entity_l1_at_0_mean": _finite_mean(l1_0s),
                "entity_l1_at_0_median": _finite_median(l1_0s),
                "entity_l1_at_end_mean": _finite_mean(l1_es),
                "entity_l1_at_end_median": _finite_median(l1_es),
                "pixel_mae_at_0_mean": _finite_mean(mae_0s),
                "pixel_mae_at_0_median": _finite_median(mae_0s),
                "pixel_mae_at_end_mean": _finite_mean(mae_es),
                "pixel_mae_at_end_median": _finite_median(mae_es),
                "per_entity_at_0": per0,
                "per_entity_at_end": pere,
                "worst_run": {
                    "traj_path": worst.get("traj_path") if worst else None,
                    "t0": worst.get("t0") if worst else None,
                    "entity_l1_at_0": (worst or {})
                    .get("metrics_at_0", {})
                    .get("entity_mean_l1"),
                    "entity_l1_at_end": (worst or {})
                    .get("metrics_at_end", {})
                    .get("entity_mean_l1"),
                    "per_entity_at_0": (worst or {}).get("per_entity_at_0") or {},
                    "per_entity_at_end": (worst or {}).get("per_entity_at_end") or {},
                }
                if worst
                else None,
            }
        )
        if not args.skip_gifs and worst is not None:
            gif_dir = os.path.join(out_dir, "worst_gifs")
            _ensure_dir(gif_dir)
            print(f"\n=== Worst-run GIF: {jax_key} ===")
            # Force CPU for GIF replay if requested.
            if args.cpu:
                os.environ.setdefault("JAX_PLATFORMS", "cpu")
            _render_worst_gif(
                jax_key=jax_key,
                run=worst,
                n=args.n,
                out_path=os.path.join(gif_dir, f"{jax_key}_worst.gif"),
                scale=args.gif_scale,
                fps=args.gif_fps,
            )

    if not args.skip_gallery:
        print("\n=== Start/end gallery ===")
        gal = [
            py,
            os.path.join(_THIS_DIR, "render_start_end_gallery.py"),
            "--corpus",
            corpus,
            "--reports-subdir",
            args.reports_subdir,
            "--out",
            os.path.join(out_dir, "start_end_gallery"),
            "--n",
            str(args.n),
            "--games",
            *games,
            *cpu,
        ]
        _run(gal)

    # Write results table (mean + median) and per-object L1
    lines = [
        "# Rebuttal micro-parity (preliminary)",
        "",
        "Protocol: OCAtari trajectory (tag=`{tag}`) → **automated** OC→JAX object "
        "transfer (no oracle / no manual latent edits) → lockstep replay of the same "
        f"actions for **n={args.n}** frames.".format(tag=args.traj_tag),
        "",
        "Primary metric: unmodified object-position L1 (player/enemies). "
        "Pixel MAE is secondary. Values are **mean / median** across sync runs.",
        "",
        "| game | L1 @0 mean | L1 @0 med | MAE @0 med | L1 @end mean | L1 @end med | MAE @end med | #syncs |",
        "|------|-----------:|----------:|-----------:|-------------:|------------:|-------------:|-------:|",
    ]
    for r in rows:
        lines.append(
            "| {game} | {a} | {b} | {c} | {d} | {e} | {f} | {n} |".format(
                game=r["game"],
                a=_fmt(r["entity_l1_at_0_mean"]),
                b=_fmt(r["entity_l1_at_0_median"]),
                c=_fmt(r["pixel_mae_at_0_median"]),
                d=_fmt(r["entity_l1_at_end_mean"]),
                e=_fmt(r["entity_l1_at_end_median"]),
                f=_fmt(r["pixel_mae_at_end_median"]),
                n=r.get("num_runs") or "—",
            )
        )

    lines += ["", "## Per-object L1 (mean / median across syncs)", ""]
    for r in rows:
        lines.append(f"### {r['game']}")
        lines.append("")
        lines.append(
            "| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |"
        )
        lines.append(
            "|--------|-----------:|----------:|-------------:|------------:|----------:|"
        )
        keys = sorted(
            set(r["per_entity_at_0"]) | set(r["per_entity_at_end"])
        )
        for k in keys:
            a = r["per_entity_at_0"].get(k) or {}
            b = r["per_entity_at_end"].get(k) or {}
            lines.append(
                "| {k} | {a0} | {a1} | {b0} | {b1} | {n} |".format(
                    k=k,
                    a0=_fmt(a.get("mean_l1")),
                    a1=_fmt(a.get("median_l1")),
                    b0=_fmt(b.get("mean_l1")),
                    b1=_fmt(b.get("median_l1")),
                    n=b.get("n_finite", 0),
                )
            )
        w = r.get("worst_run") or {}
        lines.append("")
        lines.append(
            f"Worst run: t0={w.get('t0')}  L1@0={_fmt(w.get('entity_l1_at_0'))}  "
            f"L1@end={_fmt(w.get('entity_l1_at_end'))}  "
            f"gif=`worst_gifs/{r['game']}_worst.gif`"
        )
        # Per-object on the worst run itself
        we = w.get("per_entity_at_end") or {}
        if we:
            bits = []
            for k, pe in sorted(we.items()):
                bits.append(f"{k}={_fmt((pe or {}).get('l1'))}")
            lines.append("Worst-run per-object L1@end: " + ", ".join(bits))
        lines.append("")

    lines += [
        f"Reports: `{reports_dir}`",
        f"Gallery: `{os.path.join(out_dir, 'start_end_gallery')}`",
        f"Worst GIFs: `{os.path.join(out_dir, 'worst_gifs')}`",
        "",
    ]
    results_md = os.path.join(out_dir, "results.md")
    with open(results_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    def _jsonable(x: Any) -> Any:
        if isinstance(x, dict):
            return {kk: _jsonable(vv) for kk, vv in x.items()}
        if isinstance(x, list):
            return [_jsonable(vv) for vv in x]
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            return None
        return x

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": args.n,
                "num_t0": args.num_t0,
                "traj_tag": args.traj_tag,
                "oracle": False,
                "games": _jsonable(rows),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {results_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
