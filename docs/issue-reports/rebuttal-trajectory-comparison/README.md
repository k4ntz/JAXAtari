# Rebuttal trajectory-comparison artifacts

Snapshot of **only** the human-play runs and outputs cited in
`docs/issue-reports/rebuttal-ale-jaxatari-similarity.md`.

Not part of the Sphinx site (`docs/source/`); lives next to other issue-report
notes so it will not be picked up by the docs build.

Script reference (flags only): [`scripts/oc_parity/README.md`](../../../scripts/oc_parity/README.md).

---

## What we ran for the rebuttal

Games: Freeway, Montezuma’s Revenge, Pong, Bank Heist, Seaquest, Skiing.

1. **Record** human-play OCAtari trajectories (`frameskip=1`):
   ```bash
   bash scripts/oc_parity/record_rebuttal_play.sh            # all six
   bash scripts/oc_parity/record_rebuttal_play.sh freeway    # one game
   ```
   Defaults: 100 warmup (not saved) + 1600 frames @ 30 FPS →
   `<corpus>/raw/<game>_play01.npz`.

2. **Evaluate** with automated OC→JAXtari transfer only (**no** oracle, **no**
   manual latent patches):
   ```bash
   PYTHONPATH=scripts python scripts/oc_parity/run_rebuttal_eval.py \
     --corpus data/oc_parity_corpus --cpu --n 150 --num-t0 5 --traj-tag play
   ```
   - Horizon `n=150` (~5 s @ 30 FPS), 5 random sync points per traj
   - Primary metric: object-position L1 — per object `|Δx|+|Δy|`, then mean over
     matched objects; report **mean / median across syncs** at inject and at end
   - Secondary: pixel MAE
   - Outputs: lockstep `summary.json`, `results.md` / `results.json`, worst-run
     GIFs, start/end gallery

Do **not** use `--oracle-lookahead` for reported numbers.

---

## Layout (this folder)

```
raw/                       # human-play OC trajs used in the report
reports/<game>/            # lockstep summary.json only (not strip PNG dumps)
trajectory_results/        # frozen eval outputs (not named results/ — gitignored)
  results.md               # aggregated tables (mean/median + per-object)
  results.json
  worst_gifs/              # highest L1@end sync per game
  start_end_gallery/       # best/worst start|end panels + overviews
```

## Excluded on purpose

- Random-action trajs and older `play02` / non-report games
- Full `data/oc_parity_corpus/` eval dumps (oracle reports, strip galleries,
  curve PNGs, run logs)
- Anything not used for the rebuttal numbers / visuals

---

## Reproduce from this snapshot

`run_rebuttal_eval.py` writes under `<corpus>/final_report/rebuttal/` (and a
reports subdir). This folder’s `trajectory_results/` is a frozen copy of that
output — re-run from a **writable copy**:

```bash
pyenv activate cuda_jax
cp -a docs/issue-reports/rebuttal-trajectory-comparison /tmp/rebuttal_replay
PYTHONPATH=scripts python scripts/oc_parity/run_rebuttal_eval.py \
  --corpus /tmp/rebuttal_replay \
  --reports-subdir reports \
  --cpu --n 150 --num-t0 5 --traj-tag play
# → /tmp/rebuttal_replay/final_report/rebuttal/{results.md,worst_gifs,…}
```

Record fresh human play into a corpus:

```bash
CORPUS=/tmp/rebuttal_replay bash scripts/oc_parity/record_rebuttal_play.sh
```
