#!/usr/bin/env python3
"""Batch soft lockstep eval over all implemented OC→JAX translators.

Looks for trajectories under ``--corpus/raw/<jaxkey>_*.npz`` (prefers
``*_play*.npz``, else any ``*.npz``), runs ``lockstep_eval`` settings, and
writes a cross-game ``leaderboard.json`` + console table.

Example:
  pyenv activate cuda_jax
  python scripts/oc_parity/run_all_lockstep.py \\
    --corpus data/oc_parity_corpus --cpu
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_trajs(raw_dir: str, jax_key: str) -> List[str]:
    play = sorted(glob.glob(os.path.join(raw_dir, f"{jax_key}_play*.npz")))
    if play:
        return play
    any_traj = sorted(glob.glob(os.path.join(raw_dir, f"{jax_key}_*.npz")))
    return any_traj


def _load_summary(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    agg = summary.get("aggregate", {})
    return {
        "game": summary.get("game"),
        "oc_game": summary.get("oc_game"),
        "num_runs": agg.get("num_runs"),
        "survival_soft_at_n": agg.get("survival_soft_at_n"),
        "median_first_diverge_soft": agg.get("median_first_diverge_soft"),
        "mean_entity_l1_at_0": agg.get("mean_entity_l1_at_0"),
        "mean_entity_l1_at_end": agg.get("mean_entity_l1_at_end"),
        "mean_score_abs_at_0": agg.get("mean_score_abs_at_0"),
        "mean_score_abs_at_end": agg.get("mean_score_abs_at_end"),
        "mean_mae0": agg.get("mean_mae0"),
        "mean_pixel_mae_at_0": agg.get("mean_pixel_mae_at_0"),
        "mean_pixel_mae_excess_at_end": agg.get("mean_pixel_mae_excess_at_end"),
        "mean_pixel_mae_delta_at_end": agg.get("mean_pixel_mae_delta_at_end"),
        "report_dir": None,
    }


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return "nan"
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _print_table(rows: Sequence[Dict[str, Any]], n: int) -> None:
    headers = (
        "game",
        f"surv@{n}%",
        "med_div",
        "L1@0",
        "L1@end",
        "mae0",
        "xs@end",
        "Δ@end",
        "sc@0",
    )
    print("  ".join(f"{h:>10}" for h in headers))
    print("-" * (12 * len(headers)))
    for r in rows:
        surv = r.get("survival_soft_at_n")
        surv_s = "—" if surv is None else f"{100.0 * float(surv):.1f}"
        vals = [
            str(r.get("game") or "")[:10],
            surv_s,
            _fmt(r.get("median_first_diverge_soft"), 1),
            _fmt(r.get("mean_entity_l1_at_0")),
            _fmt(r.get("mean_entity_l1_at_end")),
            _fmt(r.get("mean_mae0")),
            _fmt(r.get("mean_pixel_mae_excess_at_end")),
            _fmt(r.get("mean_pixel_mae_delta_at_end")),
            _fmt(r.get("mean_score_abs_at_0")),
        ]
        print("  ".join(f"{v:>10}" for v in vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OC→JAX soft lockstep eval")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/oc_parity_corpus",
        help="Corpus root with raw/ and reports/",
    )
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--t0-mode", choices=("auto", "fixed", "random"), default="random")
    parser.add_argument("--num-t0", type=int, default=3)
    parser.add_argument("--soft-tau", type=float, default=12.0)
    parser.add_argument("--persist", type=int, default=5)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="Optional subset of jax keys (default: all implemented)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip games with no trajs (default True)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first lockstep_eval failure",
    )
    parser.add_argument(
        "--oracle-lookahead",
        action="store_true",
        help="Pass --oracle-lookahead to each lockstep_eval",
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=None,
        help="Lookahead frames for oracle (default: same as --n)",
    )
    parser.add_argument(
        "--reports-subdir",
        type=str,
        default=None,
        help="Reports folder under corpus (default: reports, or reports_oracle with --oracle-lookahead)",
    )
    args = parser.parse_args()

    from oc_parity.translators.registry import (
        JAX_TO_OC_GAME,
        list_implemented_translators,
        print_skipped_games_note,
    )

    corpus = os.path.abspath(args.corpus)
    raw_dir = os.path.join(corpus, "raw")
    if args.reports_subdir:
        reports_name = args.reports_subdir
    elif args.oracle_lookahead:
        reports_name = "reports_oracle"
    else:
        reports_name = "reports"
    reports_dir = os.path.join(corpus, reports_name)
    os.makedirs(reports_dir, exist_ok=True)

    games = args.games or list_implemented_translators()
    print_skipped_games_note()
    print(
        f"Batch lockstep: corpus={corpus} reports={reports_name} games={len(games)} "
        f"n={args.n} t0_mode={args.t0_mode} num_t0={args.num_t0} "
        f"soft_tau={args.soft_tau} persist={args.persist} "
        f"oracle={args.oracle_lookahead}"
    )

    lockstep_py = os.path.join(_THIS_DIR, "lockstep_eval.py")
    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []

    for jax_key in games:
        trajs = _find_trajs(raw_dir, jax_key)
        if not trajs:
            msg = f"no trajectories matching {jax_key}_*.npz under {raw_dir}"
            print(f"[skip] {jax_key}: {msg}")
            errors.append((jax_key, msg))
            continue

        out_dir = os.path.join(reports_dir, jax_key)
        os.makedirs(out_dir, exist_ok=True)
        # Pass a shell-safe glob for this game's trajs.
        traj_glob = os.path.join(raw_dir, f"{jax_key}_*.npz")
        cmd = [
            sys.executable,
            lockstep_py,
            "-g",
            jax_key,
            "--trajs",
            traj_glob,
            "--n",
            str(args.n),
            "--t0-mode",
            args.t0_mode,
            "--num-t0",
            str(args.num_t0),
            "--soft-tau",
            str(args.soft_tau),
            "--persist",
            str(args.persist),
            "--tau",
            str(args.tau),
            "--seed",
            str(args.seed),
            "--out",
            out_dir,
        ]
        if args.cpu:
            cmd.append("--cpu")
        if args.oracle_lookahead:
            cmd.append("--oracle-lookahead")
            if args.oracle_horizon is not None:
                cmd.extend(["--oracle-horizon", str(args.oracle_horizon)])

        oc_name = JAX_TO_OC_GAME.get(jax_key, jax_key)
        print(f"\n=== {jax_key} (OC={oc_name}) trajs={len(trajs)} ===")
        print(" ", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=os.path.dirname(_SCRIPTS_DIR))
        if proc.returncode != 0:
            msg = f"lockstep_eval exited {proc.returncode}"
            print(f"[fail] {jax_key}: {msg}")
            errors.append((jax_key, msg))
            if args.fail_fast:
                raise SystemExit(f"fail-fast: {jax_key}: {msg}")
            continue

        summary = _load_summary(os.path.join(out_dir, "summary.json"))
        if summary is None:
            errors.append((jax_key, "missing summary.json"))
            continue
        row = _row_from_summary(summary)
        row["report_dir"] = out_dir
        row["trajs"] = [os.path.basename(p) for p in trajs]
        rows.append(row)

    # Sort by soft survival desc, then median diverge desc.
    def sort_key(r: Dict[str, Any]):
        surv = r.get("survival_soft_at_n")
        med = r.get("median_first_diverge_soft")
        return (
            -1.0 if surv is None else -float(surv),
            -1.0 if med is None else -float(med),
        )

    rows_sorted = sorted(rows, key=sort_key)
    leaderboard = {
        "corpus": corpus,
        "reports_dir": reports_dir,
        "oracle_lookahead": bool(args.oracle_lookahead),
        "n": args.n,
        "t0_mode": args.t0_mode,
        "num_t0": args.num_t0,
        "soft_tau": args.soft_tau,
        "persist": args.persist,
        "tau_strict": args.tau,
        "seed": args.seed,
        "games": rows_sorted,
        "errors": [{"game": g, "error": e} for g, e in errors],
    }
    out_lb = os.path.join(reports_dir, "leaderboard.json")
    with open(out_lb, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    print("\n=== Leaderboard (soft survival) ===")
    _print_table(rows_sorted, args.n)
    if errors:
        print("\nErrors / skips:")
        for g, e in errors:
            print(f"  - {g}: {e}")
    print(f"\nWrote {out_lb}")


if __name__ == "__main__":
    main()
