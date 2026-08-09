"""
Does the LeWM world model learn representations that are useful for control?

Runs PPO twice per game under an identical budget — once on a from-scratch CNN,
once on the frozen LeWM encoder — and plots the two return curves together. The
from-scratch arm is the control: without it, a LeWM curve on its own says nothing,
because any number could be explained by PPO rather than by the representation.

Artifacts land in one directory:

    <outdir>/
        summary.json                machine-readable results for every run
        summary.md                  the same table, ready to paste into the report
        comparison.png             per-game return curves, both arms overlaid
        <game>_<arm>_seed<n>/       per-run history.json, returns.png, agent.pt

Usage
-----
    # the overnight comparison
    python run_ppo_comparison.py --outdir results/ppo --total_timesteps 1000000

    # quick check that both arms run
    python run_ppo_comparison.py --outdir results/ppo_smoke --total_timesteps 20000
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_lewm import build_parser, train


# Pong is the simplest, Seaquest is where agents pick up shooting quickly, and
# Breakout is a standard reference point — the three the lab guidelines suggest
# starting from.
GAMES = ["pong", "seaquest", "breakout"]
ARMS = ["scratch", "lewm_frozen"]

ARM_STYLE = {
    "scratch":       dict(color="C7", linestyle="--", label="PPO from scratch"),
    "lewm_frozen":   dict(color="C0", linestyle="-",  label="PPO on frozen LeWM"),
    "lewm_finetune": dict(color="C2", linestyle="-.", label="PPO on LeWM (fine-tuned)"),
}


def build_cli():
    p = argparse.ArgumentParser(description="LeWM-vs-scratch PPO comparison")
    p.add_argument("--games", nargs="+", default=GAMES)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=list(ARM_STYLE))
    p.add_argument("--outdir", type=str, default="results/ppo")
    p.add_argument("--wm_dir", type=str, default="results/full",
                   help="world-model results; encoders read from <wm_dir>/<game>/model.pt")
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--device", type=str, default="auto")
    return p


def run_args(game, arm, cli):
    """Per-run config: ppo_lewm defaults, overridden by this comparison's CLI."""
    args = build_parser().parse_args([])          # all defaults
    args.game = game
    args.features = arm
    args.outdir = cli.outdir
    args.total_timesteps = cli.total_timesteps
    args.num_envs = cli.num_envs
    args.num_steps = cli.num_steps
    args.seed = cli.seed
    args.log_every = cli.log_every
    args.device = cli.device
    if arm != "scratch":
        args.encoder = str(Path(cli.wm_dir) / game / "model.pt")
    return args


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.empty_cache()


def curve(history):
    """(steps, returns) for iterations that actually finished an episode."""
    pts = [(e["global_step"], e["episodic_return_mean"])
           for e in history if e["episodic_return_mean"] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def smooth(y, frac=20):
    w = max(1, len(y) // frac)
    return w, np.convolve(y, np.ones(w) / w, mode="valid")


def final_return(history, last_frac=0.1):
    """Mean return over the last `last_frac` of the run — less noisy than the
    single final iteration, which often averages over one or two episodes."""
    _, rets = curve(history)
    if not rets:
        return None
    n = max(1, int(len(rets) * last_frac))
    return float(np.mean(rets[-n:]))


def save_comparison(outdir: Path, games, results):
    """The report's headline figure: both arms per game, on shared axes."""
    plotted = [g for g in games
               if any(results.get(f"{g}/{a}", {}).get("status") == "ok" for a in ARM_STYLE)]
    if not plotted:
        return None

    ncols = min(3, len(plotted))
    nrows = (len(plotted) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)

    for ax, game in zip(axes.flat, plotted):
        for arm, style in ARM_STYLE.items():
            r = results.get(f"{game}/{arm}")
            if not r or r["status"] != "ok" or not r.get("curve"):
                continue
            steps = [p[0] for p in r["curve"]]
            rets = [p[1] for p in r["curve"]]
            ax.plot(steps, rets, alpha=0.15, color=style["color"], linewidth=1)
            if len(rets) >= 4:
                w, sm = smooth(rets)
                ax.plot(steps[w - 1:], sm, **style)
            else:
                ax.plot(steps, rets, **style)
        ax.set_title(game, fontsize=11)
        ax.set_xlabel("Environment steps")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(plotted):]:
        ax.axis("off")

    axes.flat[0].set_ylabel("Episodic return")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9)
    fig.suptitle("Does a LeWM representation help PPO? (identical budget per arm)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = outdir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_summary_md(outdir: Path, games, arms, results, cli):
    lines = [
        "# LeWM representations for control — PPO comparison",
        "",
        f"`total_timesteps={cli.total_timesteps}`, `num_envs={cli.num_envs}`, "
        f"`num_steps={cli.num_steps}`, `seed={cli.seed}`",
        "",
        "Both arms share the environment, the policy/value heads, the "
        "hyperparameters and the step budget. Only the feature extractor differs, "
        "so any gap between them is attributable to the representation.",
        "",
        "`final return` is the mean over the last 10% of training, which is far "
        "less noisy than the single last iteration.",
        "",
        "| game | arm | status | final return | steps/s | time (min) |",
        "|---|---|---|---|---|---|",
    ]
    for game in games:
        for arm in arms:
            r = results.get(f"{game}/{arm}")
            if r is None:
                continue
            if r["status"] != "ok":
                lines.append(f"| {game} | {arm} | **{r['status']}** | — | — | "
                             f"{r['time_min']:.1f} |")
                continue
            fr = r["final_return"]
            lines.append(
                f"| {game} | {arm} | ok | {'—' if fr is None else f'{fr:.2f}'} "
                f"| {r['sps']} | {r['time_min']:.1f} |"
            )

    lines += ["", "## Verdict", ""]
    for game in games:
        a = results.get(f"{game}/scratch", {})
        b = results.get(f"{game}/lewm_frozen", {})
        if a.get("status") == "ok" and b.get("status") == "ok" \
                and a.get("final_return") is not None and b.get("final_return") is not None:
            d = b["final_return"] - a["final_return"]
            verdict = ("LeWM ahead" if d > 0 else "scratch ahead" if d < 0 else "tied")
            lines.append(f"- **{game}**: scratch {a['final_return']:.2f} vs "
                         f"LeWM-frozen {b['final_return']:.2f} "
                         f"(Δ {d:+.2f}) — {verdict}")
        else:
            lines.append(f"- **{game}**: incomplete")

    failed = {k: v for k, v in results.items() if v["status"] != "ok"}
    if failed:
        lines += ["", "## Failures", ""]
        for k, v in failed.items():
            lines += [f"### {k}", "", "```", v.get("error", "").strip()[-1500:], "```", ""]

    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    cli = build_cli().parse_args()
    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_file = outdir / "summary.json"

    jobs = [(g, a) for g in cli.games for a in cli.arms]

    # Merge with whatever is already in this directory, so re-running one arm
    # (say after fixing it) does not discard the arms that were fine.
    results = {}
    if summary_file.exists():
        results = json.loads(summary_file.read_text())
        keep = [k for k in results if k not in {f"{g}/{a}" for g, a in jobs}]
        if keep:
            print(f"Keeping {len(keep)} existing run(s): {', '.join(sorted(keep))}")
    run_start = time.time()

    for i, (game, arm) in enumerate(jobs, start=1):
        tag = f"{game}/{arm}"
        print(f"\n{'=' * 64}")
        print(f"[{i}/{len(jobs)}] {game} — {arm}")
        print(f"{'=' * 64}")

        args = run_args(game, arm, cli)
        if arm != "scratch" and not Path(args.encoder).exists():
            results[tag] = {"status": "missing_encoder", "time_min": 0.0,
                            "error": f"no world-model checkpoint at {args.encoder} — "
                                     f"run run_all_games.py first"}
            print(f"✗ {tag}: no encoder at {args.encoder}")
            summary_file.write_text(json.dumps(results, indent=2))
            continue

        start = time.time()
        try:
            history = train(args)
            steps, rets = curve(history)
            results[tag] = {
                "status": "ok",
                "time_min": (time.time() - start) / 60,
                "final_return": final_return(history),
                "sps": history[-1]["sps"] if history else None,
                "curve": [[s, r] for s, r in zip(steps, rets)],
            }
            print(f"✓ {tag} — final return {results[tag]['final_return']}")
        except Exception as e:
            results[tag] = {
                "status": "error",
                "time_min": (time.time() - start) / 60,
                "error": traceback.format_exc(),
            }
            print(f"✗ {tag} FAILED: {type(e).__name__}: {e}")

        free_memory()
        summary_file.write_text(json.dumps(results, indent=2))

    # Report on every arm present in the directory, not just the ones re-run.
    all_arms = [a for a in ARM_STYLE if any(k.endswith(f"/{a}") for k in results)]
    fig = save_comparison(outdir, cli.games, results)
    save_summary_md(outdir, cli.games, all_arms, results, cli)

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n{'=' * 64}")
    print(f"SUMMARY — {n_ok}/{len(results)} runs ok "
          f"in {(time.time() - run_start) / 60:.1f} min")
    print(f"{'=' * 64}")
    for tag, r in results.items():
        icon = "✓" if r["status"] == "ok" else "✗"
        extra = ""
        if r["status"] == "ok" and r["final_return"] is not None:
            extra = f" | final return {r['final_return']:7.2f}"
        print(f"{icon} {tag:28s} | {r['status']:16s} | {r['time_min']:5.1f} min{extra}")

    print(f"\nArtifacts: {outdir.resolve()}")
    print("  summary.json / summary.md" + ("\n  comparison.png" if fig else ""))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
