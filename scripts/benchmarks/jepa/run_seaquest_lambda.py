"""
Diagnose Seaquest's low effective rank: does increasing the SIGReg weight (lambda)
raise the representation rank? If yes -> tunable. If no -> data-diversity bound.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, argparse
from pathlib import Path
from lewm_jaxatari import train

LAMBDAS = [0.1, 0.5, 2.0]   # 0.1 = paper default / current baseline
OUTDIR = Path("results/lambda_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE = dict(emb_dim=256, seq_len=16, batch_size=32, lr=3e-4,
            total_steps=2500, init_sequences=200, collect_every=200,
            collect_n=10, buffer_size=10_000,
            img_size=84, seed=42, log_every=500, eval_seq=32,
            eval_context=3, device="auto", ckpt_every=0,
            resume=None, stop_grad=False)

results = {}
for lam in LAMBDAS:
    print(f"\n{'='*55}\nseaquest — lambda={lam}\n{'='*55}")
    # one outdir per lambda so the runs don't overwrite each other
    args = argparse.Namespace(game="seaquest", sigreg_weight=lam,
                              outdir=str(OUTDIR / f"lambda_{lam}"), **BASE)
    _, hist, ev = train(args)
    results[str(lam)] = {
        "final_pred_loss": hist[-1]["pred_loss"],
        "final_eff_rank": hist[-1]["eff_rank"],
        "rollout_mse_mean": ev["rollout_mse_mean"] if ev else None,
    }
    with open(OUTDIR / "seaquest_lambda_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("\n=== SEAQUEST LAMBDA SWEEP ===")
print(f"{'lambda':>8} {'pred_loss':>10} {'eff_rank':>9} {'rollout':>8}")
for lam, r in results.items():
    print(f"{lam:>8} {r['final_pred_loss']:>10.4f} {r['final_eff_rank']:>9.1f} {r['rollout_mse_mean']:>8.3f}")
