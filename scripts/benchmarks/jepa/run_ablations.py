"""
Ablations: which parts of LeWM actually do the work?

Each variant changes exactly one thing against the faithful configuration and is
trained identically otherwise, so a difference in the metrics is attributable to
that change.

    faithful      the paper: AdaLN actions, no stop-gradient, lambda = 0.1
    no_sigreg     lambda = 0 — removes the anti-collapse term entirely
    stop_grad     stop-gradient on the target, on top of SIGReg
    stop_grad_only  stop-gradient with lambda = 0 — the BYOL-style control
    action_add    additive action embedding instead of AdaLN
    sigreg_0.5    lambda = 0.5
    sigreg_2.0    lambda = 2.0

The ones that matter most are `no_sigreg`, `stop_grad` and `stop_grad_only`,
because together they test the paper's central claim: that SIGReg alone prevents
representation collapse, with no stop-gradient and no EMA target encoder.

`no_sigreg` shows SIGReg is needed *given no stop-gradient*. `stop_grad_only` is
the harder question and the real BYOL-style control: with lambda = 0 and a
stop-gradient, does the asymmetry alone hold the representation open? Without it
one can only claim SIGReg is sufficient, not that it does anything a plain
stop-gradient could not — and this predictor is already asymmetric enough that
the question is live.

Effective rank is the collapse metric rather than embedding variance: BatchNorm
pins per-dimension variance to 1, so collapse shows up as the embeddings
occupying a low-dimensional subspace, not as shrinking norms.

Usage
-----
    python run_ablations.py --outdir results/ablations --total_steps 5000
    python run_ablations.py --outdir results/abl_smoke --total_steps 200 \
        --init_sequences 48 --games pong
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

from lewm_jaxatari import DEFAULTS, train


VARIANTS = {
    "faithful":   dict(),
    "no_sigreg":  dict(sigreg_weight=0.0),
    "stop_grad":  dict(stop_grad=True),
    "stop_grad_only": dict(sigreg_weight=0.0, stop_grad=True),
    "action_add": dict(action_cond="add"),
    "sigreg_0.5": dict(sigreg_weight=0.5),
    "sigreg_2.0": dict(sigreg_weight=2.0),
}

STYLE = {
    "faithful":   dict(color="C0", linestyle="-",  linewidth=2),
    "no_sigreg":  dict(color="C3", linestyle="-",  linewidth=2),
    "stop_grad":  dict(color="C1", linestyle="--"),
    "stop_grad_only": dict(color="C6", linestyle="--", linewidth=2),
    "action_add": dict(color="C2", linestyle="-."),
    "sigreg_0.5": dict(color="C4", linestyle=":"),
    "sigreg_2.0": dict(color="C5", linestyle=":"),
}


def build_cli():
    p = argparse.ArgumentParser(description="LeWM ablations")
    p.add_argument("--games", nargs="+", default=["pong", "seaquest"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS))
    p.add_argument("--outdir", type=str, default="results/ablations")
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--init_sequences", type=int, default=600)
    p.add_argument("--collect_every", type=int, default=50)
    p.add_argument("--collect_n", type=int, default=20)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p


def variant_args(game, variant, cli):
    cfg = dict(DEFAULTS)
    cfg.update(
        game=game,
        outdir=str(Path(cli.outdir) / variant),
        total_steps=cli.total_steps,
        init_sequences=cli.init_sequences,
        collect_every=cli.collect_every,
        collect_n=cli.collect_n,
        log_every=cli.log_every,
        eval_seq=cli.eval_seq,
        seed=cli.seed,
        device=cli.device,
        plot=True,
        plot_every=0,          # the per-variant curve is drawn once at the end
    )
    cfg.update(VARIANTS[variant])
    return argparse.Namespace(**cfg)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.empty_cache()


def save_figure(outdir: Path, games, variants, results):
    """Effective rank (collapse) and prediction loss, per game, all variants."""
    rows = [g for g in games
            if any(results.get(f"{g}/{v}", {}).get("status") == "ok" for v in variants)]
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.4 * len(rows)), squeeze=False)
    for r, game in enumerate(rows):
        ax_rank, ax_pred = axes[r]
        for v in variants:
            res = results.get(f"{game}/{v}")
            if not res or res["status"] != "ok":
                continue
            steps = res["steps"]
            ax_rank.plot(steps, res["eff_rank"], label=v, **STYLE.get(v, {}))
            ax_pred.plot(steps, res["pred_loss"], label=v, **STYLE.get(v, {}))
        ax_rank.set_title(f"{game}: effective rank (collapse → 1)", fontsize=10)
        ax_rank.set_ylabel("effective rank")
        ax_rank.set_ylim(bottom=0)
        ax_pred.set_title(f"{game}: prediction loss", fontsize=10)
        ax_pred.set_ylabel("pred loss")
        ax_pred.set_yscale("log")
        for ax in (ax_rank, ax_pred):
            ax.set_xlabel("training step")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        ax_rank.legend(fontsize=7)

    fig.suptitle("LeWM ablations — does SIGReg alone prevent collapse?", fontsize=12)
    fig.tight_layout()
    path = outdir / "ablations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_summary_md(outdir: Path, games, variants, results, cli):
    lines = [
        "# LeWM ablations",
        "",
        f"`total_steps={cli.total_steps}`, `seed={cli.seed}`. Each variant changes "
        "exactly one thing against `faithful`.",
        "",
        "`eff. rank` is the effective rank of the embeddings out of 256 — near 1 "
        "means representation collapse. `rollout MSE` is open-loop latent "
        "prediction error against a `frozen` baseline that assumes the embedding "
        "never changes; a model that does not beat it has learned nothing useful.",
        "",
        "| game | variant | pred loss | eff. rank | rollout MSE | frozen | beats |",
        "|---|---|---|---|---|---|---|",
    ]
    for game in games:
        for v in variants:
            r = results.get(f"{game}/{v}")
            if r is None:
                continue
            if r["status"] != "ok":
                lines.append(f"| {game} | {v} | **{r['status']}** | — | — | — | — |")
                continue
            roll, base = r["rollout_mse_mean"], r["frozen_baseline_mean"]
            beats = "—" if roll is None or base is None else ("yes" if roll < base else "**no**")
            lines.append(
                f"| {game} | {v} | {r['final_pred_loss']:.4f} | {r['final_eff_rank']:.1f} "
                f"| {'—' if roll is None else f'{roll:.4f}'} "
                f"| {'—' if base is None else f'{base:.4f}'} | {beats} |"
            )

    lines += ["", "## The paper's central claim", ""]
    for game in games:
        f = results.get(f"{game}/faithful", {})
        n = results.get(f"{game}/no_sigreg", {})
        s = results.get(f"{game}/stop_grad", {})
        if f.get("status") != "ok":
            continue
        parts = [f"**{game}**: faithful holds effective rank "
                 f"{f['final_eff_rank']:.1f}/256 with no stop-gradient"]
        if n.get("status") == "ok":
            parts.append(f"removing SIGReg gives {n['final_eff_rank']:.1f}")
        if s.get("status") == "ok":
            parts.append(f"adding stop-gradient gives {s['final_eff_rank']:.1f}")
        lines.append("- " + "; ".join(parts) + ".")

    failed = {k: v for k, v in results.items() if v["status"] != "ok"}
    if failed:
        lines += ["", "## Failures", ""]
        for k, v in failed.items():
            lines += [f"### {k}", "", "```", v.get("error", "").strip()[-1200:], "```", ""]

    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    cli = build_cli().parse_args()
    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_file = outdir / "summary.json"

    jobs = [(g, v) for g in cli.games for v in cli.variants]
    results, run_start = {}, time.time()

    for i, (game, variant) in enumerate(jobs, start=1):
        tag = f"{game}/{variant}"
        print(f"\n{'=' * 64}")
        print(f"[{i}/{len(jobs)}] {tag}")
        print(f"{'=' * 64}")

        start = time.time()
        try:
            _, hist, ev = train(variant_args(game, variant, cli))
            results[tag] = {
                "status": "ok",
                "time_min": (time.time() - start) / 60,
                "final_pred_loss": hist[-1]["pred_loss"],
                "final_eff_rank": hist[-1]["eff_rank"],
                "rollout_mse_mean": ev["rollout_mse_mean"] if ev else None,
                "frozen_baseline_mean": ev["frozen_baseline_mean"] if ev else None,
                "steps": [e["step"] for e in hist],
                "eff_rank": [e["eff_rank"] for e in hist],
                "pred_loss": [e["pred_loss"] for e in hist],
            }
            print(f"✓ {tag} — eff_rank {results[tag]['final_eff_rank']:.1f}, "
                  f"pred {results[tag]['final_pred_loss']:.4f}")
        except Exception as e:
            results[tag] = {"status": "error", "time_min": (time.time() - start) / 60,
                            "error": traceback.format_exc()}
            print(f"✗ {tag} FAILED: {type(e).__name__}: {e}")

        free_memory()
        summary_file.write_text(json.dumps(results, indent=2))

    fig = save_figure(outdir, cli.games, cli.variants, results)
    save_summary_md(outdir, cli.games, cli.variants, results, cli)

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n{'=' * 64}")
    print(f"SUMMARY — {n_ok}/{len(results)} ok in {(time.time() - run_start) / 60:.1f} min")
    print(f"{'=' * 64}")
    for tag, r in results.items():
        icon = "✓" if r["status"] == "ok" else "✗"
        extra = (f" | eff_rank {r['final_eff_rank']:5.1f} | pred {r['final_pred_loss']:.4f}"
                 if r["status"] == "ok" else "")
        print(f"{icon} {tag:26s} | {r['time_min']:5.1f} min{extra}")

    print(f"\nArtifacts: {outdir.resolve()}")
    print("  summary.json / summary.md" + ("\n  ablations.png" if fig else ""))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
