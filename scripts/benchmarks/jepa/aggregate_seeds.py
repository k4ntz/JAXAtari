"""
Aggregate repeated agent runs across seeds.

A single PPO run on Atari says very little: seed-to-seed spread on Pong is larger
than most effects worth reporting. This script pools several `run_ppo_comparison`
output directories that differ only in `--seed` and reports, per game and arm:

  * final return as mean +- standard deviation over seeds
  * steps to first positive mean episodic return (sample efficiency), which is
    what a pretrained representation is actually supposed to improve
  * a learning curve per arm with a band spanning the seeds

The distinction matters. On Pong the *final* returns of PPO-from-scratch and
PPO-on-frozen-LeWM overlap heavily once seeds are taken into account, while the
steps-to-positive-play do not overlap at all, so the defensible claim is about
sample efficiency and run-to-run consistency, not about the final score.

Usage
-----
    python aggregate_seeds.py --dirs results/ppo results/ppo_s2 results/ppo_s3 \
        --game pong --out results/seeds
"""

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_STYLE = {
    "scratch":       dict(color="0.35", linestyle="--", label="PPO from scratch"),
    "lewm_frozen":   dict(color="C0",   linestyle="-",  label="PPO on frozen LeWM"),
    "lewm_finetune": dict(color="C2",   linestyle="-.", label="PPO on LeWM (fine-tuned)"),
}


def load_runs(dirs, game, arm):
    """Every (seed, final_return, history) triple for this game/arm.

    `final_return` comes from the runner's summary, which averages the last 10%
    of iterations. The last history entry is a single noisy evaluation, often
    `None`, and using it silently collapses the seed spread.
    """
    runs = []
    for d in dirs:
        summary = Path(d) / "summary.json"
        if not summary.exists():
            continue
        entry = json.loads(summary.read_text()).get(f"{game}/{arm}", {})
        if entry.get("status") != "ok":
            continue
        for run_dir in sorted(Path(d).glob(f"{game}_{arm}_seed*")):
            hist = json.loads((run_dir / "history.json").read_text())
            runs.append((int(run_dir.name.rsplit("seed", 1)[1]),
                         entry.get("final_return"), hist))
    return runs


def steps_to_positive(history):
    """First environment step at which the mean episodic return goes above zero.

    Returns None if it never does, reported rather than silently dropped, since
    "never got there" is itself a result.
    """
    for e in history["history"]:
        r = e.get("episodic_return_mean")
        if r is not None and r > 0:
            return e["global_step"]
    return None


def curve_band(runs, n_points=200):
    """Mean and min/max envelope of the seeds on a shared step grid."""
    series = []
    for _, _, hist in runs:
        pts = [(e["global_step"], e["episodic_return_mean"])
               for e in hist["history"] if e["episodic_return_mean"] is not None]
        if len(pts) > 1:
            series.append(np.array(pts).T)
    if not series:
        return None
    hi = min(s[0].max() for s in series)
    grid = np.linspace(min(s[0].min() for s in series), hi, n_points)
    stacked = np.stack([np.interp(grid, s[0], s[1]) for s in series])
    return grid, stacked.mean(0), stacked.min(0), stacked.max(0)


def main():
    p = argparse.ArgumentParser(description="Pool agent runs over seeds")
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--game", default="pong")
    p.add_argument("--arms", nargs="+", default=list(ARM_STYLE))
    p.add_argument("--out", default="results/seeds")
    cli = p.parse_args()

    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, fig, ax = [], *plt.subplots(figsize=(7, 4.2))
    for arm in cli.arms:
        runs = load_runs(cli.dirs, cli.game, arm)
        if not runs:
            continue

        finals = [f for _, f, _ in runs if f is not None]
        firsts = [steps_to_positive(h) for _, _, h in runs]

        rows.append({
            "arm": arm,
            "n_seeds": len(runs),
            "final_mean": stats.mean(finals) if finals else None,
            "final_sd": stats.stdev(finals) if len(finals) > 1 else 0.0,
            "finals": finals,
            "steps_to_positive": firsts,
            "steps_to_positive_mean": (stats.mean([f for f in firsts if f])
                                       if any(firsts) else None),
        })

        band = curve_band(runs)
        if band:
            g, mean, lo, hi = band
            style = ARM_STYLE.get(arm, {})
            ax.plot(g, mean, linewidth=2, **style)
            ax.fill_between(g, lo, hi, alpha=0.18, color=style.get("color"))

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"{cli.game}: mean over {rows[0]['n_seeds']} seeds, band = min/max"
                 if rows else cli.game)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{cli.game}_seeds.png", dpi=150)
    plt.close(fig)

    (out / f"{cli.game}_seeds.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{cli.game}, {rows[0]['n_seeds'] if rows else 0} seeds\n")
    print(f"{'arm':16s} {'final return':>20s} {'steps to positive':>22s}")
    for r in rows:
        fin = f"{r['final_mean']:.2f} +- {r['final_sd']:.2f}" if r["final_mean"] is not None else "n/a"
        stp = (f"{r['steps_to_positive_mean']/1e6:.2f}M "
               f"{[round(f/1e6, 2) if f else None for f in r['steps_to_positive']]}"
               if r["steps_to_positive_mean"] else "never")
        print(f"{r['arm']:16s} {fin:>20s} {stp:>22s}")
    print(f"\nWrote {out}/{cli.game}_seeds.png and .json")


if __name__ == "__main__":
    main()
