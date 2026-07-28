# `oc_parity/` scripts

Requires `PYTHONPATH=scripts` (and usually `pyenv activate cuda_jax`). Optional: `ocatari`.

Each script also supports `--help`.

---

## `record_ocatari.py`

Record one OCAtari trajectory (actions + objects + RAM + frames) to `.npz`.

| Arg | Description |
|-----|-------------|
| `-g` / `--game` | Game name |
| `--mode {random,play}` | Random actions or interactive pygame |
| `--steps` | Frames to record after warmup |
| `--skip-frames` | Warmup frames stepped but not saved (default 500) |
| `--seed` | RNG seed |
| `--out` | Output `.npz` path (**required**) |
| `--no-frames` | Omit RGB frames |
| `--fps` | Play-mode playback FPS |
| `--summarize` | Print summary after save |

---

## `record_corpus.py`

Batch-record trajs for translators into `<corpus>/raw/<jaxkey>_<tag>.npz`.

| Arg | Description |
|-----|-------------|
| `--corpus` | Corpus root (default `data/oc_parity_corpus`) |
| `--mode {random,play}` | Recording mode |
| `--skip-frames` / `--steps` / `--seed` | Warmup, length, seed |
| `--games` | Optional subset of JAX keys (default: all implemented) |
| `--force` | Overwrite existing files |
| `--tag` | Filename tag (default `play01`) |

---

## `record_rebuttal_play.sh`

Shell wrapper around `record_ocatari.py` for Freeway, MontezumaRevenge, Pong, BankHeist, Seaquest, Skiing.

| Usage / env | Description |
|-------------|-------------|
| *(no args)* | Record all six games in order (Enter between) |
| `<jax_key>` | Record one game (e.g. `freeway`) |
| `print` | Print the underlying `record_ocatari.py` commands only |
| `CORPUS` | Corpus root (default `data/oc_parity_corpus`) |
| `SKIP_FRAMES` | Warmup (default `100`) |
| `STEPS` | Recorded frames (default `1600`) |
| `TAG` | Filename tag (default `play01`) |
| `FPS` | Play FPS (default `30`) |

---

## `lockstep_eval.py`

Inject OC→JAX at sync frame(s) `t0`, replay `--n` actions, write a report dir.

| Arg | Description |
|-----|-------------|
| `-g` / `--game` | JAX game key |
| `--trajs` | Trajectory path or glob (**required**; single glob string) |
| `--n` | Lockstep horizon (frames) |
| `--t0-mode {auto,fixed,random}` | How to choose sync points |
| `--t0` | Fixed sync index (`t0-mode=fixed`) |
| `--num-t0` | Random syncs per traj (`t0-mode=random`) |
| `--skip-frames` | Only allow `t0` ≥ this index |
| `--tau` | Strict entity L∞ threshold (diagnostic) |
| `--soft-tau` | Soft mean entity L1 threshold (survival) |
| `--persist` | Consecutive soft breaches before diverge |
| `--seed` | RNG for random `t0` |
| `--out` | Report output directory (**required**) |
| `--cpu` | Force JAX CPU |
| `--max-trajs` | Cap number of traj files |
| `--oracle-lookahead` | Bake OC-future spawns/moves into JAX latents |
| `--oracle-horizon` | Lookahead frames for oracle (default: `--n`) |

---

## `run_rebuttal_eval.py`

Batch lockstep over a game list: automated transfer only (no oracle), tables, worst-run GIFs, optional gallery.

| Arg | Description |
|-----|-------------|
| `--corpus` | Corpus root |
| `--games` | JAX keys (default: freeway, montezumarevenge, pong, bankheist, seaquest, skiing) |
| `--n` | Lockstep horizon |
| `--num-t0` | Random syncs per traj |
| `--seed` | RNG seed |
| `--skip-frames` / `--steps` | Used only with `--record` |
| `--traj-tag` | Traj filename prefix (default `play` → `<game>_play*.npz`) |
| `--record` | Record random trajs before eval |
| `--force-record` | Overwrite when recording |
| `--reports-subdir` | Lockstep report folder under corpus (default `reports_rebuttal`) |
| `--cpu` | Force JAX CPU |
| `--skip-gallery` | Skip start/end PNG gallery |
| `--skip-gifs` | Skip worst-run GIFs |
| `--gif-fps` / `--gif-scale` | Worst-GIF render settings |

Outputs under `<corpus>/final_report/rebuttal/` and `<corpus>/<reports-subdir>/`.

---

## `run_all_lockstep.py`

Batch `lockstep_eval.py` over all (or selected) translators + leaderboard.

| Arg | Description |
|-----|-------------|
| `--corpus` | Corpus with `raw/` |
| `--n` / `--t0-mode` / `--num-t0` | Lockstep sync settings |
| `--soft-tau` / `--persist` / `--tau` | Survival thresholds |
| `--seed` / `--cpu` | RNG / device |
| `--games` | Optional JAX-key subset |
| `--skip-missing` | Skip games with no trajs |
| `--fail-fast` | Abort on first failure |
| `--oracle-lookahead` / `--oracle-horizon` | Pass through to lockstep |
| `--reports-subdir` | Reports folder (default `reports`, or `reports_oracle` with oracle) |

---

## `build_final_report.py`

Merge baseline/oracle lockstep reports into a packaged `final_report/`.

| Arg | Description |
|-----|-------------|
| `--corpus` | Corpus root |
| `--baseline-dir` / `--oracle-dir` | Input report trees |
| `--out` | Output directory |
| `--problem-diverge` / `--problem-l1` / `--problem-excess` | Problem-gallery thresholds |

---

## `render_start_end_gallery.py`

Best/worst start\|end `OC|JAX|diff` PNG panels from lockstep summaries.

| Arg | Description |
|-----|-------------|
| `--corpus` | Corpus root |
| `--reports-subdir` | Where summaries live |
| `--out` | Output dir |
| `--n` / `--scale` | Horizon / pixel scale |
| `--cpu` | Force JAX CPU |
| `--games` | Optional game list |

---

## `render_comparison_videos.py`

Per-game best vs worst soft-survival GIFs (`OC|JAX|diff` strips).

| Arg | Description |
|-----|-------------|
| `--corpus` / `--reports-subdir` | Inputs |
| `--out` | Output dir |
| `--games` | Optional subset |
| `--n` | Frames to render (default: from summary) |
| `--fps` / `--scale` | GIF settings |
| `--oracle-lookahead` | Replay with oracle inject |
| `--cpu` | Force JAX CPU |
| `--max-games` | Cap for smoke tests |

---

## `oracle_lookahead.py`

Library helper (also used via `--oracle-lookahead` on lockstep). Bakes OC-future spawn/movement into JAX latent state for supported games.

---

## Supporting modules

| Module | Role |
|--------|------|
| `trajectory_io.py` | Load/save `.npz` trajectories |
| `metrics.py` | Entity L1 / pixel / survival metrics |
| `report.py` | Write summaries + plots |
| `translators/` | Per-game OC→JAXtari state + compare extractors (`registry.py`) |

---

## Related

- Broader scripts index: [`scripts/README.md`](../README.md)
- Frozen rebuttal assets + protocol notes: [`docs/issue-reports/rebuttal-trajectory-comparison/`](../../docs/issue-reports/rebuttal-trajectory-comparison/)
