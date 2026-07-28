#!/usr/bin/env python3
"""Offline lockstep OC→JAX micro-parity evaluation.

Injects an OC snapshot at one or more sync frames ``t0``, replays the same
actions on JAX for ``n`` frames, and scores entity / score / pixel agreement.

Examples:
  # Record a few trajs first, then:
  python scripts/oc_parity/lockstep_eval.py \\
    --game Pong --trajs '/tmp/pong_*.npz' --n 240 --t0-mode random --num-t0 3 \\
    --out /tmp/pong_lockstep --cpu

  python scripts/oc_parity/lockstep_eval.py \\
    --game BankHeist --trajs '/tmp/bh.npz' --n 120 --t0-mode fixed --t0 0 \\
    --out /tmp/bh_lockstep --cpu
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Force JAX on CPU before importing jax when requested.
if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _map_oc_action_to_jax(env, oc_action: int, meanings: Optional[Sequence[str]]) -> Any:
    """Map an OC/ALE action index onto the JAX env's ACTION_SET index."""
    import jax.numpy as jnp
    from jaxatari.environment import JAXAtariAction

    action_set = np.asarray(
        getattr(env, "ACTION_SET", None)
        if getattr(env, "ACTION_SET", None) is not None
        else getattr(env, "action_set")
    )

    if meanings is not None and 0 <= int(oc_action) < len(meanings):
        name = str(meanings[int(oc_action)]).upper()
        const = getattr(JAXAtariAction, name, None)
        if const is not None:
            matches = np.where(action_set == int(const))[0]
            if len(matches):
                return jnp.asarray(int(matches[0]), dtype=jnp.int32)

    # Fallback: assume identical compact indexing.
    idx = int(oc_action)
    if idx < 0 or idx >= len(action_set):
        matches = np.where(action_set == int(JAXAtariAction.NOOP))[0]
        idx = int(matches[0]) if len(matches) else 0
    return jnp.asarray(idx, dtype=jnp.int32)


def _valid_t0_candidates(
    traj: Dict[str, Any],
    jax_key: str,
    n: int,
    *,
    skip_frames: int = 0,
) -> List[int]:
    T = len(traj["actions"])
    max_t0 = T - n
    if max_t0 < 0:
        return []
    start = int(max(0, skip_frames))
    # Prefer traj meta skip if caller left default 0 but recording already stripped warmup.
    # (Saved trajs with skip_frames already omitted those frames, so start stays 0.)
    if jax_key == "pong":
        from oc_parity.trajectory_io import find_first_ball_frame

        ball_t = find_first_ball_frame(traj)
        if ball_t is not None:
            start = max(start, int(ball_t))
    if start > max_t0:
        return []
    return list(range(start, max_t0 + 1))


def _sample_t0s(
    candidates: Sequence[int],
    *,
    mode: str,
    fixed_t0: Optional[int],
    num_t0: int,
    rng: np.random.Generator,
) -> List[int]:
    if not candidates:
        return []
    if mode == "fixed":
        t0 = 0 if fixed_t0 is None else int(fixed_t0)
        if t0 not in candidates:
            # Clamp to nearest valid
            t0 = min(candidates, key=lambda c: abs(c - t0))
        return [t0]
    if mode == "auto":
        return [candidates[0]]
    # random: sample without replacement up to num_t0
    k = min(int(num_t0), len(candidates))
    chosen = rng.choice(np.asarray(candidates), size=k, replace=False)
    return sorted(int(x) for x in chosen)


def _inject_state(env, jax_key: str, traj: Dict[str, Any], t0: int):
    import inspect

    from oc_parity.translators.registry import get_translator
    from oc_parity.translators.base import objects_as_dicts, find_object

    translate = get_translator(jax_key)
    objs = traj["objects"][t0]
    seed = int(traj.get("meta", {}).get("seed", 0))
    meanings = traj.get("meta", {}).get("action_meanings")
    kwargs: Dict[str, Any] = {
        "seed": seed,
        "frame_index": t0,
        "action_meanings": meanings,
        "oc_action": int(traj["actions"][t0]) if traj.get("actions") is not None else None,
        "prev_oc_action": (
            int(traj["actions"][t0 - 1])
            if t0 > 0 and traj.get("actions") is not None
            else None
        ),
    }

    prev_xy = None
    prev_player = None
    if t0 > 0:
        # Prefer Player; Freeway uses Chicken.
        prev_dicts = objects_as_dicts(traj["objects"][t0 - 1])
        prev_player = find_object(prev_dicts, "Player")
        if prev_player is None:
            prev_player = find_object(prev_dicts, "Chicken")
        if prev_player is not None:
            prev_xy = (float(prev_player["x"]), float(prev_player["y"]))

    # Multi-frame player XY lookback (BankHeist coasts on alternate frames).
    lookback_n = 16
    xy_lookback: List[Tuple[float, float]] = []
    t_lo = max(0, t0 - lookback_n + 1)
    for t in range(t_lo, t0 + 1):
        d = objects_as_dicts(traj["objects"][t])
        p = find_object(d, "Player")
        if p is None:
            p = find_object(d, "Chicken")
        if p is not None and int(p.get("w", 1)) > 0:
            xy_lookback.append((float(p["x"]), float(p["y"])))
    kwargs["player_xy_lookback"] = xy_lookback

    kwargs["prev_player_y"] = None if prev_xy is None else prev_xy[1]
    kwargs["prev_player_xy"] = prev_xy
    kwargs["prev_enemy_y"] = None
    if t0 > 0:
        prev_enemy = find_object(objects_as_dicts(traj["objects"][t0 - 1]), "Enemy")
        if prev_enemy is not None:
            kwargs["prev_enemy_y"] = float(prev_enemy["y"])
        kwargs["prev_objects"] = traj["objects"][t0 - 1]
    else:
        kwargs["prev_objects"] = None

    # Translators differ in signature; only pass accepted kwargs (or **kwargs).
    sig = inspect.signature(translate)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return translate(env, objs, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return translate(env, objs, **filtered)


def _extract_pair(jax_key: str, env, state, oc_objects):
    import importlib

    mod = importlib.import_module(f"oc_parity.translators.{jax_key}")
    extract_oc = getattr(mod, "extract_oc_compare_entities", None)
    extract_jax = getattr(mod, "extract_jax_compare_entities", None)
    if extract_oc is None or extract_jax is None:
        raise SystemExit(f"No compare extractors for {jax_key}")

    try:
        oc_cmp = extract_oc(oc_objects, consts=getattr(env, "consts", None))
    except TypeError:
        oc_cmp = extract_oc(oc_objects)
    return oc_cmp, extract_jax(state, env)


def run_one_lockstep(
    env,
    jitted_step,
    jitted_render,
    traj: Dict[str, Any],
    *,
    jax_key: str,
    t0: int,
    n: int,
    tau: float,
    soft_tau: float,
    persist: int,
    keep_strips: bool,
    strip_ks: Sequence[int],
    oracle_lookahead: bool = False,
    oracle_horizon: Optional[int] = None,
) -> Dict[str, Any]:
    from oc_parity.metrics import (
        baseline_relative_pixel_metrics,
        change_maps,
        combine_frame_metrics,
        entity_metrics,
        first_diverge_index,
        first_diverge_persistent,
        pixel_metrics,
    )

    meanings = traj.get("meta", {}).get("action_meanings")
    state = _inject_state(env, jax_key, traj, t0)
    if oracle_lookahead:
        from oc_parity.oracle_lookahead import apply_oracle_lookahead, supports_oracle

        if supports_oracle(jax_key):
            state = apply_oracle_lookahead(
                env,
                state,
                traj,
                jax_key=jax_key,
                t0=t0,
                horizon=int(oracle_horizon if oracle_horizon is not None else n),
            )

    series: Dict[str, List[float]] = {
        "entity_mean_l1": [],
        "entity_max_linf": [],
        "entity_presence": [],
        "score_mean_abs": [],
        "score_max_abs": [],
        "pixel_mae": [],
        "pixel_equal_frac": [],
        "pixel_mae_excess": [],
        "pixel_mae_delta": [],
        "pixel_mae_signed_delta": [],
    }
    strict_flags: List[bool] = []
    soft_flags: List[bool] = []
    strip_frames: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * n
    change_strip_frames: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
        None
    ] * n
    # Full timeline for motion heatmaps (t0 → soft-diverge / end).
    motion_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * n

    oc0: Optional[np.ndarray] = None
    jax0: Optional[np.ndarray] = None
    mae0: Optional[float] = None
    per_entity_at_0: Optional[Dict[str, Any]] = None
    per_entity_at_end: Optional[Dict[str, Any]] = None

    for k in range(n):
        t = t0 + k
        oc_cmp, jax_cmp = _extract_pair(jax_key, env, state, traj["objects"][t])
        ent = entity_metrics(oc_cmp, jax_cmp, tau=tau, soft_tau=soft_tau)
        if k == 0:
            per_entity_at_0 = ent.get("per_entity") or {}
        if k == n - 1:
            per_entity_at_end = ent.get("per_entity") or {}

        jax_frame = np.asarray(jitted_render(state), dtype=np.uint8)
        if traj["frames"] is not None:
            oc_frame = np.asarray(traj["frames"][t], dtype=np.uint8)
            if k == 0:
                oc0 = oc_frame.copy()
                jax0 = jax_frame.copy()
                mae0 = float(pixel_metrics(oc0, jax0)["pixel_mae"])
                pix = baseline_relative_pixel_metrics(
                    oc_frame, jax_frame, oc0, jax0, mae0=mae0
                )
            else:
                assert oc0 is not None and jax0 is not None
                pix = baseline_relative_pixel_metrics(
                    oc_frame, jax_frame, oc0, jax0, mae0=mae0
                )
            if keep_strips:
                motion_pairs[k] = (oc_frame.copy(), jax_frame.copy())
                if k in strip_ks:
                    strip_frames[k] = motion_pairs[k]
                    if oc0 is not None and jax0 is not None:
                        change_strip_frames[k] = change_maps(
                            oc_frame, jax_frame, oc0, jax0
                        )
        else:
            pix = {
                "pixel_mae": float("nan"),
                "pixel_mae_norm": float("nan"),
                "pixel_equal_frac": float("nan"),
                "pixel_mae0": float("nan"),
                "pixel_mae_excess": float("nan"),
                "pixel_mae_delta": float("nan"),
                "pixel_mae_signed_delta": float("nan"),
            }

        frame = combine_frame_metrics(ent, pix)
        for key in series:
            series[key].append(float(frame.get(key, float("nan"))))
        strict_flags.append(bool(frame["strict_breach"]))
        soft_flags.append(bool(frame["soft_breach"]))

        if k < n - 1:
            oc_a = int(traj["actions"][t])
            jax_a = _map_oc_action_to_jax(env, oc_a, meanings)
            _o, state, _r, _d, _i = jitted_step(state, jax_a)

    fd_strict = first_diverge_index(strict_flags)
    fd_soft = first_diverge_persistent(soft_flags, persist=persist)
    return {
        "t0": t0,
        "n": n,
        "traj_seed": traj.get("meta", {}).get("seed"),
        # Headline = soft (dynamics-tolerant)
        "first_diverge": fd_soft,
        "survived": fd_soft is None,
        "first_diverge_soft": fd_soft,
        "survived_soft": fd_soft is None,
        "first_diverge_strict": fd_strict,
        "survived_strict": fd_strict is None,
        "series": series,
        "strip_frames": strip_frames if keep_strips else None,
        "change_strip_frames": change_strip_frames if keep_strips else None,
        "motion_pairs": motion_pairs if keep_strips else None,
        "mae0": mae0,
        "metrics_at_0": {k: series[k][0] for k in series},
        "metrics_at_end": {k: series[k][-1] for k in series},
        "per_entity_at_0": per_entity_at_0 or {},
        "per_entity_at_end": per_entity_at_end or {},
    }


def aggregate_runs(runs: Sequence[Dict[str, Any]], n: int) -> Dict[str, Any]:
    from oc_parity.metrics import survival_rate

    fds_soft = [r["first_diverge_soft"] for r in runs]
    fds_strict = [r["first_diverge_strict"] for r in runs]
    surv_soft = survival_rate(fds_soft, n)
    surv_strict = survival_rate(fds_strict, n)

    def stack_mean(key):
        arr = np.asarray([r["series"][key] for r in runs], dtype=np.float64)
        return {
            "mean": np.nanmean(arr, axis=0).tolist(),
            "q25": np.nanpercentile(arr, 25, axis=0).tolist(),
            "q75": np.nanpercentile(arr, 75, axis=0).tolist(),
        }

    def _stat_fds(fds):
        finite = [n if fd is None else fd for fd in fds]
        return {
            "median": float(np.median(finite)) if finite else None,
            "mean": float(np.mean(finite)) if finite else None,
            "values": fds,
        }

    soft_stats = _stat_fds(fds_soft)
    strict_stats = _stat_fds(fds_strict)
    return {
        "num_runs": len(runs),
        "n": n,
        # Headline soft metrics
        "survival_at_n": surv_soft,
        "survival_soft_at_n": surv_soft,
        "survival_strict_at_n": surv_strict,
        "median_first_diverge": soft_stats["median"],
        "mean_first_diverge": soft_stats["mean"],
        "median_first_diverge_soft": soft_stats["median"],
        "mean_first_diverge_soft": soft_stats["mean"],
        "median_first_diverge_strict": strict_stats["median"],
        "mean_first_diverge_strict": strict_stats["mean"],
        "mean_entity_l1_at_0": float(
            np.nanmean([r["metrics_at_0"]["entity_mean_l1"] for r in runs])
        ),
        "mean_entity_l1_at_end": float(
            np.nanmean([r["metrics_at_end"]["entity_mean_l1"] for r in runs])
        ),
        "mean_score_abs_at_0": float(
            np.nanmean([r["metrics_at_0"]["score_mean_abs"] for r in runs])
        ),
        "mean_score_abs_at_end": float(
            np.nanmean([r["metrics_at_end"]["score_mean_abs"] for r in runs])
        ),
        "mean_pixel_mae_at_0": float(
            np.nanmean([r["metrics_at_0"]["pixel_mae"] for r in runs])
        ),
        "mean_pixel_mae_at_end": float(
            np.nanmean([r["metrics_at_end"]["pixel_mae"] for r in runs])
        ),
        "mean_pixel_mae_excess_at_end": float(
            np.nanmean([r["metrics_at_end"].get("pixel_mae_excess", np.nan) for r in runs])
        ),
        "mean_pixel_mae_delta_at_end": float(
            np.nanmean([r["metrics_at_end"].get("pixel_mae_delta", np.nan) for r in runs])
        ),
        "mean_mae0": float(
            np.nanmean(
                [
                    r.get("mae0", r["metrics_at_0"].get("pixel_mae", np.nan))
                    for r in runs
                ]
            )
        ),
        "curves": {
            "entity_mean_l1": stack_mean("entity_mean_l1"),
            "score_mean_abs": stack_mean("score_mean_abs"),
            "pixel_mae": stack_mean("pixel_mae"),
            "pixel_mae_excess": stack_mean("pixel_mae_excess"),
            "pixel_mae_delta": stack_mean("pixel_mae_delta"),
        },
        "first_diverges": fds_soft,
        "first_diverges_soft": fds_soft,
        "first_diverges_strict": fds_strict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OC→JAX lockstep micro-parity eval")
    parser.add_argument("-g", "--game", type=str, required=True)
    parser.add_argument(
        "--trajs",
        type=str,
        required=True,
        help="Trajectory path or glob, e.g. '/tmp/pong_*.npz'",
    )
    parser.add_argument("--n", type=int, default=240, help="Lockstep horizon (frames)")
    parser.add_argument(
        "--t0-mode",
        choices=("auto", "fixed", "random"),
        default="random",
        help="auto=first valid; fixed=--t0; random=sample --num-t0 sync points",
    )
    parser.add_argument("--t0", type=int, default=None, help="Fixed sync frame (t0-mode=fixed)")
    parser.add_argument(
        "--num-t0",
        type=int,
        default=3,
        help="How many random t0 sync points per trajectory (t0-mode=random)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Only allow t0 >= this index (use for old trajs that still include spawn). "
        "New recordings with --skip-frames already omit warmup, so leave 0.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Strict entity L∞ threshold (px); diagnostic only",
    )
    parser.add_argument(
        "--soft-tau",
        type=float,
        default=12.0,
        help="Soft mean entity L1 threshold (px); headline survival uses this",
    )
    parser.add_argument(
        "--persist",
        type=int,
        default=5,
        help="Consecutive soft-breach frames required before counting as diverged",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random t0 sampling")
    parser.add_argument("--out", type=str, required=True, help="Report output directory")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=None,
        help="Optional cap on number of trajectory files",
    )
    parser.add_argument(
        "--oracle-lookahead",
        action="store_true",
        help="After inject, bake OC-future spawn/movement into JAX latent state "
        "(seaquest/enduro/asteroids/bankheist/mspacman/venture/beamrider/phoenix)",
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=None,
        help="Lookahead frames for oracle (default: same as --n)",
    )
    args = parser.parse_args()

    import jax
    import jaxatari
    from oc_parity.trajectory_io import load_trajectory
    from oc_parity.translators.registry import (
        list_implemented_translators,
        normalize_game_name,
        print_disclaimers,
        print_skipped_games_note,
    )
    from oc_parity import report as report_mod

    jax_key, oc_name = normalize_game_name(args.game)
    if jax_key not in list_implemented_translators():
        raise SystemExit(
            f"No translator for '{jax_key}'. Implemented: {list_implemented_translators()}"
        )

    paths = sorted(glob.glob(os.path.expanduser(args.trajs)))
    if not paths and os.path.isfile(os.path.expanduser(args.trajs)):
        paths = [os.path.expanduser(args.trajs)]
    if not paths:
        raise SystemExit(f"No trajectories matched: {args.trajs}")
    if args.max_trajs is not None:
        paths = paths[: args.max_trajs]

    print_skipped_games_note()
    print_disclaimers(jax_key)
    print(
        f"Lockstep eval: game={jax_key} (OC={oc_name}) n={args.n} "
        f"t0_mode={args.t0_mode} num_t0={args.num_t0} skip_frames={args.skip_frames} "
        f"soft_tau={args.soft_tau} persist={args.persist} tau_strict={args.tau} "
        f"oracle={args.oracle_lookahead} trajs={len(paths)}"
    )

    env = jaxatari.make(jax_key)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)
    # Warm JIT
    _o, st = env.reset(jax.random.PRNGKey(0))
    _ = jitted_render(st)
    _ = jitted_step(st, _map_oc_action_to_jax(env, 0, None))

    rng = np.random.default_rng(args.seed)
    strip_ks = [0, max(args.n // 4, 1), max(args.n // 2, 1), max((3 * args.n) // 4, 1), args.n - 1]
    strip_ks = sorted(set(k for k in strip_ks if 0 <= k < args.n))

    runs: List[Dict[str, Any]] = []
    for path in paths:
        traj = load_trajectory(path)
        if traj["frames"] is None:
            print(f"Warning: {path} has no frames; pixel metrics will be NaN")
        cands = _valid_t0_candidates(
            traj, jax_key, args.n, skip_frames=args.skip_frames
        )
        if not cands:
            print(
                f"Skip {path}: need T >= skip_frames({args.skip_frames}) + n({args.n}) "
                f"(have T={len(traj['actions'])})"
            )
            continue
        t0s = _sample_t0s(
            cands,
            mode=args.t0_mode,
            fixed_t0=args.t0,
            num_t0=args.num_t0,
            rng=rng,
        )
        print(f"  {os.path.basename(path)}: {len(cands)} valid t0s, using {t0s}")
        for t0 in t0s:
            print(f"  run {os.path.basename(path)} t0={t0} ...")
            run = run_one_lockstep(
                env,
                jitted_step,
                jitted_render,
                traj,
                jax_key=jax_key,
                t0=t0,
                n=args.n,
                tau=args.tau,
                soft_tau=args.soft_tau,
                persist=args.persist,
                keep_strips=True,
                strip_ks=strip_ks,
                oracle_lookahead=args.oracle_lookahead,
                oracle_horizon=args.oracle_horizon,
            )
            run["traj_path"] = path
            runs.append(run)
            fd_s = run["first_diverge_soft"]
            fd_h = run["first_diverge_strict"]
            print(
                f"    soft_diverge={fd_s if fd_s is not None else 'survived'} "
                f"(strict={fd_h if fd_h is not None else 'survived'}) "
                f"entity_l1@0={run['metrics_at_0']['entity_mean_l1']:.2f} "
                f"pixel_mae@0={run['metrics_at_0']['pixel_mae']:.2f} "
                f"mae_excess@end={run['metrics_at_end'].get('pixel_mae_excess', float('nan')):.2f} "
                f"score_abs@0={run['metrics_at_0']['score_mean_abs']:.2f}"
            )

    if not runs:
        raise SystemExit("No successful lockstep runs")

    agg = aggregate_runs(runs, args.n)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "game": jax_key,
        "oc_game": oc_name,
        "n": args.n,
        "tau_strict": args.tau,
        "soft_tau": args.soft_tau,
        "persist": args.persist,
        "t0_mode": args.t0_mode,
        "num_t0": args.num_t0,
        "skip_frames": args.skip_frames,
        "oracle_lookahead": args.oracle_lookahead,
        "oracle_horizon": args.oracle_horizon if args.oracle_horizon is not None else args.n,
        "aggregate": {k: v for k, v in agg.items() if k != "curves"},
        "runs": [
            {
                "traj_path": r["traj_path"],
                "t0": r["t0"],
                "traj_seed": r["traj_seed"],
                "first_diverge_soft": r["first_diverge_soft"],
                "survived_soft": r["survived_soft"],
                "first_diverge_strict": r["first_diverge_strict"],
                "survived_strict": r["survived_strict"],
                "metrics_at_0": r["metrics_at_0"],
                "metrics_at_end": r["metrics_at_end"],
                "per_entity_at_0": r.get("per_entity_at_0") or {},
                "per_entity_at_end": r.get("per_entity_at_end") or {},
            }
            for r in runs
        ],
    }
    report_mod.write_summary_json(os.path.join(out_dir, "summary.json"), summary)

    # Plots (vline = soft median diverge)
    report_mod.plot_time_series(
        os.path.join(out_dir, "entity_l1.png"),
        {"mean": agg["curves"]["entity_mean_l1"]["mean"],
         "q25": agg["curves"]["entity_mean_l1"]["q25"],
         "q75": agg["curves"]["entity_mean_l1"]["q75"]},
        title=f"{jax_key}: entity mean L1 (soft_tau={args.soft_tau})",
        vline=agg["median_first_diverge_soft"],
    )
    report_mod.plot_time_series(
        os.path.join(out_dir, "score_abs.png"),
        {"mean": agg["curves"]["score_mean_abs"]["mean"],
         "q25": agg["curves"]["score_mean_abs"]["q25"],
         "q75": agg["curves"]["score_mean_abs"]["q75"]},
        title=f"{jax_key}: score mean |Δ|",
        vline=agg["median_first_diverge_soft"],
    )
    report_mod.plot_time_series(
        os.path.join(out_dir, "pixel_mae.png"),
        {"mean": agg["curves"]["pixel_mae"]["mean"],
         "q25": agg["curves"]["pixel_mae"]["q25"],
         "q75": agg["curves"]["pixel_mae"]["q75"]},
        title=f"{jax_key}: pixel MAE (absolute)",
        vline=agg["median_first_diverge_soft"],
    )
    report_mod.plot_time_series(
        os.path.join(out_dir, "pixel_mae_excess.png"),
        {"mean": agg["curves"]["pixel_mae_excess"]["mean"],
         "q25": agg["curves"]["pixel_mae_excess"]["q25"],
         "q75": agg["curves"]["pixel_mae_excess"]["q75"]},
        title=f"{jax_key}: pixel MAE excess vs transfer (mae - mae0)",
        vline=agg["median_first_diverge_soft"],
    )
    report_mod.plot_time_series(
        os.path.join(out_dir, "pixel_mae_delta.png"),
        {"mean": agg["curves"]["pixel_mae_delta"]["mean"],
         "q25": agg["curves"]["pixel_mae_delta"]["q25"],
         "q75": agg["curves"]["pixel_mae_delta"]["q75"]},
        title=f"{jax_key}: change-map MAE (|OC-oc0| vs |JAX-jax0|)",
        vline=agg["median_first_diverge_soft"],
    )
    report_mod.plot_diverge_histogram(
        os.path.join(out_dir, "first_diverge_hist.png"),
        agg["first_diverges_soft"],
        args.n,
        title=f"Soft first-diverge (mean L1>{args.soft_tau} for {args.persist} frames)",
    )
    report_mod.plot_diverge_histogram(
        os.path.join(out_dir, "first_diverge_strict_hist.png"),
        agg["first_diverges_strict"],
        args.n,
        title=f"Strict first-diverge (L∞>{args.tau} any frame)",
    )

    reps = report_mod.select_representative_runs(
        runs, args.n, key_field="first_diverge_soft"
    )
    for label, run in reps.items():
        report_mod.write_run_strips(out_dir, label, run, strip_ks)

    print("\n=== Aggregate ===")
    print(
        f"runs={agg['num_runs']}  "
        f"Survival_soft@{args.n}={agg['survival_soft_at_n']*100:.1f}%  "
        f"Survival_strict@{args.n}={agg['survival_strict_at_n']*100:.1f}%"
    )
    print(
        f"median soft_diverge={agg['median_first_diverge_soft']:.1f}  "
        f"median strict_diverge={agg['median_first_diverge_strict']:.1f}  "
        f"entity_l1 @0={agg['mean_entity_l1_at_0']:.2f} @end={agg['mean_entity_l1_at_end']:.2f}"
    )
    print(
        f"score |Δ| @0={agg['mean_score_abs_at_0']:.2f} @end={agg['mean_score_abs_at_end']:.2f}  "
        f"pixel MAE @0={agg['mean_pixel_mae_at_0']:.2f} @end={agg['mean_pixel_mae_at_end']:.2f}  "
        f"mae0={agg['mean_mae0']:.2f} excess@end={agg['mean_pixel_mae_excess_at_end']:.2f} "
        f"delta@end={agg['mean_pixel_mae_delta_at_end']:.2f}"
    )
    print(f"Report written to {out_dir}")


if __name__ == "__main__":
    main()
