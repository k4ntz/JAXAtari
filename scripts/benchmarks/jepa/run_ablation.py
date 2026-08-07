"""
Ablation: faithful LeWM (no stop-grad) vs. stop-gradient, on a few games.
Demonstrates LeWM's central claim — SIGReg prevents collapse WITHOUT stop-grad.

Artifacts go to results/ablation/: <variant>/<game>/ per training run, plus a
faithful-vs-stop_grad comparison plot and ablation_results.json at the top.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, argparse, traceback
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lewm_jaxatari import train

GAMES = ["pong", "seaquest"]
OUTDIR = Path("results/ablation")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE = dict(emb_dim=256, seq_len=16, batch_size=32, lr=3e-4,
            total_steps=4000, init_sequences=200, collect_every=200,
            collect_n=10, buffer_size=10_000, sigreg_weight=0.1,
            img_size=84, seed=42, log_every=200, eval_seq=32,
            eval_context=3, device="auto", ckpt_every=0, resume=None)

results = {}
for game in GAMES:
    results[game] = {}
    histories = {}
    for tag, sg in [("faithful", False), ("stop_grad", True)]:
        print(f"\n{'='*55}\n{game} — {tag}\n{'='*55}")
        # separate outdir per variant so the two runs don't overwrite each other
        args = argparse.Namespace(game=game, stop_grad=sg,
                                  outdir=str(OUTDIR / tag), **BASE)
        try:
            _, hist, ev = train(args)
            histories[tag] = hist
            results[game][tag] = {
                "final_pred_loss": hist[-1]["pred_loss"],
                "final_eff_rank": hist[-1]["eff_rank"],
                "rollout_mse_mean": ev["rollout_mse_mean"] if ev else None,
                "frozen_baseline_mean": ev["frozen_baseline_mean"] if ev else None,
            }
        except Exception as e:
            results[game][tag] = {"error": traceback.format_exc()[-400:]}
            print("FAILED:", e)
    # comparison plot: eff_rank over steps, faithful vs stop_grad
    if all(t in histories for t in ("faithful", "stop_grad")):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for tag, sty in [("faithful", "-"), ("stop_grad", "--")]:
            h = histories[tag]
            st = [e["step"] for e in h]
            a1.plot(st, [e["eff_rank"] for e in h], sty, label=tag)
            a2.plot(st, [e["pred_loss"] for e in h], sty, label=tag)
        a1.set_title(f"{game}: effective rank (collapse)"); a1.set_xlabel("step"); a1.set_ylabel("eff_rank"); a1.legend(); a1.grid(alpha=.3)
        a2.set_title(f"{game}: prediction loss"); a2.set_xlabel("step"); a2.set_ylabel("pred_loss"); a2.legend(); a2.grid(alpha=.3)
        plot_path = OUTDIR / f"ablation_{game}.png"
        fig.tight_layout(); fig.savefig(plot_path, dpi=150); plt.close(fig)
        print(f"  plot -> {plot_path}")
    with open(OUTDIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("\n=== ABLATION SUMMARY ===")
for g, d in results.items():
    for tag, r in d.items():
        if "error" in r: print(f"{g:10s} {tag:10s} FAILED"); continue
        print(f"{g:10s} {tag:10s} pred={r['final_pred_loss']:.4f} rank={r['final_eff_rank']:6.1f} rollout={r['rollout_mse_mean']:.3f}")
