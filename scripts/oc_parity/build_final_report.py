#!/usr/bin/env python3
"""Assemble a final OC→JAX parity report from baseline (+ optional oracle) runs.

Reads ``reports/`` and optionally ``reports_oracle/`` under the corpus, writes:

  final_report/
    summary.md
    leaderboard_baseline.json
    leaderboard_oracle.json   (if present)
    comparison.json
    figures/{game}_*.png      (copied curves)
    problems/{game}_worst_/   (endpoint diff + motion heatmap + meta)

Problem dump rule: soft diverge < --problem-diverge OR L1@0 > --problem-l1
OR mae_excess@end > --problem-excess (worst run only).

Example:
  PYTHONPATH=scripts python scripts/oc_parity/build_final_report.py \\
    --corpus data/oc_parity_corpus
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
        if x != x:
            return "nan"
        if abs(x) == float("inf"):
            return "inf"
        return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _triage_tag(row: Dict[str, Any], oracle_row: Optional[Dict[str, Any]]) -> str:
    l1 = row.get("mean_entity_l1_at_0")
    med = row.get("median_first_diverge_soft")
    mae0 = row.get("mean_mae0")
    excess = row.get("mean_pixel_mae_excess_at_end")
    try:
        l1f = float(l1) if l1 is not None else float("nan")
    except (TypeError, ValueError):
        l1f = float("nan")
    try:
        medf = float(med) if med is not None else 0.0
    except (TypeError, ValueError):
        medf = 0.0
    try:
        mae0f = float(mae0) if mae0 is not None else 0.0
        exf = float(excess) if excess is not None else 0.0
    except (TypeError, ValueError):
        mae0f, exf = 0.0, 0.0

    if l1f != l1f or l1f > 5.0:  # NaN or high
        return "mapping_bug"
    if mae0f > 20 and abs(exf) < 2:
        return "static_render_only"
    o_med = None
    if oracle_row is not None:
        try:
            o_med = float(oracle_row.get("median_first_diverge_soft") or 0)
        except (TypeError, ValueError):
            o_med = None
    if medf >= 100:
        return "strong"
    if o_med is not None and o_med > medf + 20:
        return "rng_ceiling"
    if medf < 30:
        return "mechanics_or_latent"
    return "solid_mid"


def _is_problem(
    summary: Dict[str, Any],
    *,
    diverge_lt: float,
    l1_gt: float,
    excess_gt: float,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    runs = summary.get("runs") or []
    if not runs:
        return False, None
    n = int(summary.get("n", 240))

    def score(r):
        fd = r.get("first_diverge_soft")
        return n + 1 if fd is None else int(fd)

    worst = min(runs, key=score)
    fd = worst.get("first_diverge_soft")
    fd_v = n if fd is None else int(fd)
    l1 = float(worst.get("metrics_at_0", {}).get("entity_mean_l1", 0) or 0)
    if l1 != l1:
        l1 = 1e9
    excess = float(worst.get("metrics_at_end", {}).get("pixel_mae_excess", 0) or 0)
    bad = fd_v < diverge_lt or l1 > l1_gt or excess > excess_gt
    return bad, worst


def _copy_if(src: str, dst: str) -> bool:
    if not os.path.isfile(src):
        return False
    _ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final OC→JAX parity report")
    parser.add_argument("--corpus", type=str, default="data/oc_parity_corpus")
    parser.add_argument("--baseline-dir", type=str, default="reports")
    parser.add_argument("--oracle-dir", type=str, default="reports_oracle")
    parser.add_argument("--out", type=str, default="final_report")
    parser.add_argument("--problem-diverge", type=float, default=60.0)
    parser.add_argument("--problem-l1", type=float, default=5.0)
    parser.add_argument("--problem-excess", type=float, default=2.0)
    args = parser.parse_args()

    corpus = os.path.abspath(args.corpus)
    base_root = os.path.join(corpus, args.baseline_dir)
    ora_root = os.path.join(corpus, args.oracle_dir)
    out_root = os.path.join(corpus, args.out)
    _ensure_dir(out_root)
    _ensure_dir(os.path.join(out_root, "figures"))
    _ensure_dir(os.path.join(out_root, "problems"))

    base_lb = _load_json(os.path.join(base_root, "leaderboard.json"))
    ora_lb = _load_json(os.path.join(ora_root, "leaderboard.json"))
    if base_lb is None:
        raise SystemExit(f"Missing baseline leaderboard: {base_root}/leaderboard.json")

    if base_lb:
        with open(os.path.join(out_root, "leaderboard_baseline.json"), "w", encoding="utf-8") as f:
            json.dump(base_lb, f, indent=2)
    if ora_lb:
        with open(os.path.join(out_root, "leaderboard_oracle.json"), "w", encoding="utf-8") as f:
            json.dump(ora_lb, f, indent=2)

    ora_by_game = {
        r["game"]: r for r in (ora_lb or {}).get("games", []) if r.get("game")
    }

    comparison: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []
    lines: List[str] = []
    lines.append("# OC→JAX micro-parity final report\n")
    lines.append(f"Corpus: `{corpus}`\n")
    lines.append(
        f"Settings: n={base_lb.get('n')} t0={base_lb.get('t0_mode')}/"
        f"{base_lb.get('num_t0')} soft_tau={base_lb.get('soft_tau')} "
        f"persist={base_lb.get('persist')}\n"
    )
    lines.append("\n## Leaderboard (baseline)\n")
    lines.append(
        "| game | tag | med_div | L1@0 | mae0 | excess@end | oracle_med_div |\n"
        "|------|-----|---------|------|------|------------|----------------|\n"
    )

    for row in base_lb.get("games", []):
        game = row["game"]
        ora = ora_by_game.get(game)
        tag = _triage_tag(row, ora)
        o_med = ora.get("median_first_diverge_soft") if ora else None
        lines.append(
            f"| {game} | {tag} | {_fmt(row.get('median_first_diverge_soft'), 1)} | "
            f"{_fmt(row.get('mean_entity_l1_at_0'))} | {_fmt(row.get('mean_mae0'))} | "
            f"{_fmt(row.get('mean_pixel_mae_excess_at_end'))} | {_fmt(o_med, 1)} |\n"
        )
        comparison.append(
            {
                "game": game,
                "tag": tag,
                "baseline": row,
                "oracle": ora,
            }
        )

        # Copy curve figures
        gdir = os.path.join(base_root, game)
        for name in (
            "entity_l1.png",
            "pixel_mae_excess.png",
            "pixel_mae_delta.png",
            "first_diverge_hist.png",
        ):
            _copy_if(
                os.path.join(gdir, name),
                os.path.join(out_root, "figures", f"{game}_{name}"),
            )

        summary = _load_json(os.path.join(gdir, "summary.json"))
        if summary is None:
            continue
        bad, worst = _is_problem(
            summary,
            diverge_lt=args.problem_diverge,
            l1_gt=args.problem_l1,
            excess_gt=args.problem_excess,
        )
        if not bad or worst is None:
            continue

        fd = worst.get("first_diverge_soft")
        t0 = worst.get("t0")
        prob_dir = os.path.join(out_root, "problems", f"{game}_worst_t{t0}")
        _ensure_dir(prob_dir)
        # Prefer worst strip folder from lockstep
        src_worst = os.path.join(gdir, "strips", "worst")
        copied = []
        if os.path.isdir(src_worst):
            for fn in os.listdir(src_worst):
                if fn.endswith(".png"):
                    if _copy_if(os.path.join(src_worst, fn), os.path.join(prob_dir, fn)):
                        copied.append(fn)
        meta = {
            "game": game,
            "tag": tag,
            "t0": t0,
            "first_diverge_soft": fd,
            "metrics_at_0": worst.get("metrics_at_0"),
            "metrics_at_end": worst.get("metrics_at_end"),
            "traj_path": worst.get("traj_path"),
            "copied_assets": copied,
        }
        with open(os.path.join(prob_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        problems.append(meta)

    with open(os.path.join(out_root, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"games": comparison, "problems": problems}, f, indent=2)

    lines.append("\n## Tag legend\n")
    lines.append(
        "- **strong** — low inject error, soft survival ≳ 100 frames\n"
        "- **solid_mid** — good inject, moderate soft horizon\n"
        "- **rng_ceiling** — oracle extends survival vs baseline\n"
        "- **mechanics_or_latent** — early soft diverge even with good inject\n"
        "- **mapping_bug** — high / non-finite entity L1 at inject\n"
        "- **static_render_only** — high mae0 but flat excess (palette/HUD)\n"
    )
    lines.append("\n## Problem galleries\n")
    if not problems:
        lines.append("No runs met the problem thresholds.\n")
    else:
        lines.append(
            f"Thresholds: soft_diverge < {args.problem_diverge} OR "
            f"L1@0 > {args.problem_l1} OR excess@end > {args.problem_excess}\n\n"
        )
        for p in problems:
            rel = f"problems/{p['game']}_worst_t{p['t0']}"
            lines.append(
                f"### {p['game']} (t0={p['t0']}, soft_div={p['first_diverge_soft']}, "
                f"tag={p['tag']})\n"
            )
            lines.append(f"Assets in `{rel}/` — look for:\n")
            lines.append(
                "- `k000.png` — OC | JAX | |diff| at inject\n"
                "- `*_endpoint.png` — absolute diff at soft-diverge / end\n"
                "- `motion_t0_to_k*.png` — OC-motion | JAX-motion | mismatch heatmap\n"
                "- `k*_change.png` — change-from-inject maps\n\n"
            )

    lines.append("\n## How to regenerate\n")
    lines.append("```bash\n")
    lines.append(
        "PYTHONPATH=scripts python scripts/oc_parity/run_all_lockstep.py "
        "--corpus data/oc_parity_corpus --cpu\n"
        "PYTHONPATH=scripts python scripts/oc_parity/run_all_lockstep.py "
        "--corpus data/oc_parity_corpus --cpu --oracle-lookahead "
        "--reports-subdir reports_oracle\n"
        "PYTHONPATH=scripts python scripts/oc_parity/build_final_report.py "
        "--corpus data/oc_parity_corpus\n"
    )
    lines.append("```\n")

    summary_path = os.path.join(out_root, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"Wrote {summary_path}")
    print(f"Games: {len(comparison)}  Problems: {len(problems)}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
