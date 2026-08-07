"""
Run LeWM training on all 15 required JAXtari games and collect the results.

Runs in-process (no subprocess) so JAX only pays its tracing/compilation cost
once per game. Every artifact lands under one directory:

    <outdir>/
        summary.json          machine-readable results for all games
        summary.md            the same table, ready to paste into the report
        all_games.png         grid of every learning curve
        <game>/curve.png      per-game learning curve (written during training)
        <game>/history.json   per-game loss history + eval metrics
        <game>/model.pt       final encoder + predictor weights

A game that crashes is recorded with its traceback and the run continues, so one
broken environment never costs you the other fourteen.

Examples
--------
    # quick check that all 15 games train (~1k steps each)
    python run_all_games.py --outdir results/smoke --total_steps 1000

    # the real run
    python run_all_games.py --outdir results/full --total_steps 10000 \
        --init_sequences 500 --log_every 100
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
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

import torch

from lewm_jaxatari import DEFAULTS, train


# The 15 games required by the Praktikum.
GAMES = [
    "asteroids", "beamrider", "breakout", "enduro", "freeway",
    "frostbite", "gravitar", "kangaroo", "montezumarevenge", "mspacman",
    "phoenix", "pong", "seaquest", "skiing", "tennis",
]


def build_parser():
    p = argparse.ArgumentParser(description="Train LeWM on every required JAXtari game")
    p.add_argument("--games", nargs="+", default=GAMES,
                   help="subset of games to run (default: all 15)")
    p.add_argument("--outdir", type=str, default="results/smoke",
                   help="directory for all artifacts")
    p.add_argument("--total_steps", type=int, default=1000)
    p.add_argument("--init_sequences", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--collect_every", type=int, default=DEFAULTS["collect_every"])
    p.add_argument("--collect_n", type=int, default=DEFAULTS["collect_n"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--sigreg_weight", type=float, default=DEFAULTS["sigreg_weight"])
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--plot_every", type=int, default=250)
    p.add_argument("--eval_seq", type=int, default=32)
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ckpt_every", type=int, default=0)
    p.add_argument("--stop_grad", action="store_true",
                   help="ABLATION ONLY — run every game with stop-gradient")
    return p


def game_args(game: str, cli) -> argparse.Namespace:
    """Per-game training config: package defaults, overridden by this run's CLI."""
    cfg = dict(DEFAULTS)
    cfg.update(
        game=game,
        outdir=cli.outdir,
        total_steps=cli.total_steps,
        init_sequences=cli.init_sequences,
        seq_len=cli.seq_len,
        batch_size=cli.batch_size,
        collect_every=cli.collect_every,
        collect_n=cli.collect_n,
        lr=cli.lr,
        sigreg_weight=cli.sigreg_weight,
        log_every=cli.log_every,
        plot_every=cli.plot_every,
        eval_seq=cli.eval_seq,
        seed=cli.seed,
        device=cli.device,
        ckpt_every=cli.ckpt_every,
        stop_grad=cli.stop_grad,
        plot=True,
    )
    return argparse.Namespace(**cfg)


def free_memory():
    """Drop the finished game's model before building the next one."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.empty_cache()


def save_grid(outdir: Path, games, results):
    """One figure with every game's learning curve — the report's overview plot."""
    ok = [g for g in games if results.get(g, {}).get("status") == "ok"
          and results[g].get("curve_points")]
    if not ok:
        return None

    ncols = 3
    nrows = (len(ok) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows),
                             squeeze=False)
    for ax, game in zip(axes.flat, ok):
        pts = results[game]["curve_points"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="C0", linewidth=1.5,
                label="total")
        ax.plot([p[0] for p in pts], [p[2] for p in pts], color="C1", linestyle="--",
                linewidth=1, label="pred")
        ax.set_title(game, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(ok):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.supxlabel("Training step", fontsize=9)
    fig.supylabel("Loss", fontsize=9)
    fig.suptitle("LeWM on JAXtari — learning curves", fontsize=12)
    fig.tight_layout()
    path = outdir / "all_games.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_summary_md(outdir: Path, results, cli):
    """Markdown table of the run — paste-ready for the report."""
    lines = [
        "# LeWM on JAXtari — run summary",
        "",
        f"`total_steps={cli.total_steps}`, `seq_len={cli.seq_len}`, "
        f"`batch_size={cli.batch_size}`, `lr={cli.lr}`, "
        f"`sigreg_weight={cli.sigreg_weight}`, `seed={cli.seed}`",
        "",
        "`rollout MSE` is the open-loop latent prediction error averaged over the "
        "horizon; `frozen` is the trivial baseline that assumes the embedding never "
        "changes. Lower rollout MSE than frozen means the predictor is doing real work. "
        "`eff. rank` is the effective rank of the embeddings (out of 256) — a value "
        "near 1 would mean representation collapse.",
        "",
        "| game | status | time (s) | final loss | final pred | eff. rank | rollout MSE | frozen | beats baseline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v, spec=".4f"):
        return "—" if v is None else format(v, spec)

    for game, r in results.items():
        if r["status"] != "ok":
            lines.append(f"| {game} | **{r['status']}** | {r.get('time_seconds', '—')} "
                         f"| — | — | — | — | — | — |")
            continue
        roll, base = r["rollout_mse_mean"], r["frozen_baseline_mean"]
        beats = "—" if roll is None or base is None else ("yes" if roll < base else "no")
        lines.append(
            f"| {game} | ok | {r['time_seconds']:.0f} | {fmt(r['final_loss'])} "
            f"| {fmt(r['final_pred_loss'])} | {fmt(r['final_eff_rank'], '.1f')} "
            f"| {fmt(roll)} | {fmt(base)} | {beats} |"
        )

    failed = [g for g, r in results.items() if r["status"] != "ok"]
    if failed:
        lines += ["", "## Failures", ""]
        for g in failed:
            lines += [f"### {g}", "", "```", results[g].get("error", "").strip(), "```", ""]

    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    cli = build_parser().parse_args()
    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_file = outdir / "summary.json"

    results = {}
    run_start = time.time()

    for i, game in enumerate(cli.games, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(cli.games)}] Training on: {game}")
        print(f"{'=' * 60}")

        args = game_args(game, cli)
        start = time.time()

        try:
            model, loss_history, eval_metrics = train(args)
            elapsed = time.time() - start
            last = loss_history[-1] if loss_history else {}
            results[game] = {
                "status": "ok",
                "time_seconds": round(elapsed, 1),
                "final_loss": last.get("loss"),
                "final_pred_loss": last.get("pred_loss"),
                "final_sigreg_loss": last.get("sigreg_loss"),
                "final_eff_rank": last.get("eff_rank"),
                "rollout_mse_mean": eval_metrics["rollout_mse_mean"] if eval_metrics else None,
                "frozen_baseline_mean": eval_metrics["frozen_baseline_mean"] if eval_metrics else None,
                # compact (step, loss, pred_loss) triples for the overview grid
                "curve_points": [[e["step"], e["loss"], e["pred_loss"]] for e in loss_history],
            }
            print(f"✓ {game} done in {elapsed:.0f}s")
            del model
        except Exception as e:
            elapsed = time.time() - start
            results[game] = {
                "status": "error",
                "time_seconds": round(elapsed, 1),
                "error": traceback.format_exc(),
            }
            print(f"✗ {game} FAILED after {elapsed:.0f}s: {type(e).__name__}: {e}")

        free_memory()

        # Persist after every game — a run interrupted at game 12 keeps 11 results.
        summary_file.write_text(json.dumps(results, indent=2))

    grid = save_grid(outdir, cli.games, results)
    save_summary_md(outdir, results, cli)

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {n_ok}/{len(results)} games trained "
          f"in {(time.time() - run_start) / 60:.1f} min")
    print(f"{'=' * 60}")
    for game, r in results.items():
        icon = "✓" if r["status"] == "ok" else "✗"
        extra = ""
        if r["status"] == "ok" and r["final_loss"] is not None:
            extra = f" | loss {r['final_loss']:.4f} | eff_rank {r['final_eff_rank']:.1f}"
        print(f"{icon} {game:18s} | {r['status']:6s} | {r.get('time_seconds', '?'):>7}s{extra}")

    print(f"\nArtifacts:  {outdir.resolve()}")
    print(f"  summary.json / summary.md")
    if grid:
        print(f"  all_games.png")
    print(f"  <game>/curve.png, <game>/history.json, <game>/model.pt")

    # Non-zero exit if anything failed, so CI or a shell loop notices.
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
