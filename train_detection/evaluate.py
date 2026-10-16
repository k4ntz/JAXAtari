"""Evaluate detector checkpoints end-to-end: YOLOObjectCentricWrapper observations vs. the game state.

For every observation field the wrapper fills, converted objects are matched (Hungarian, top-left corner
within --hit-px game pixels) to the visible ground-truth objects of ObjectCentricWrapper for the same state.
Reports recall, precision and position error per field, plus wrapper throughput (jitted, vmapped).

    python train_detection/evaluate.py gray96/pong rgb640/pong          # checkpoint names or .npz paths
    python train_detection/evaluate.py rgb640/pong rgb640/breakout --steps 200 --batch 1 --json results.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util
from scipy.optimize import linear_sum_assignment

import jaxtari
from jaxtari.detection import MISSING, YOLOObjectCentricWrapper, load_detector
from jaxtari.wrappers import AtariWrapper


def _visible(oo, screen_w, screen_h):
    p = {k: np.atleast_1d(np.asarray(getattr(oo, k))).ravel() for k in ("x", "y", "width", "height", "active")}
    m = (p["active"] > 0) & (p["x"] > 1) & (p["y"] > 1) & (p["width"] > 0) & (p["height"] > 0) & \
        (p["x"] < screen_w) & (p["y"] < screen_h)  # off-screen objects (unclipped coordinates) are not visible
    return np.stack([p["x"][m], p["y"][m]], 1)


def _match(gt, pred, hit_px):
    if len(gt) == 0 or len(pred) == 0:
        return []
    d = np.linalg.norm(gt[:, None] - pred[None], axis=-1)
    r, c = linear_sum_assignment(d)
    return [d[i, j] for i, j in zip(r, c) if d[i, j] <= hit_px]


def evaluate(path, steps=200, batch_sizes=(1, 128), hit_px=3.0, max_pool=False, seed=1):
    det = load_detector(path)
    env = YOLOObjectCentricWrapper(AtariWrapper(jaxtari.make(det.game)), det, max_pool=max_pool)
    base = env._env._env
    screen_h, screen_w = base.renderer.config.game_dimensions
    as_float = jax.tree.map(lambda a: np.asarray(a, np.float32), env._template)
    _, unravel = flatten_util.ravel_pytree(as_float)
    gt_flat = jax.jit(lambda es: flatten_util.ravel_pytree(base._get_observation(es))[0].astype(jnp.float32))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    obs, st = reset(k)
    stats = defaultdict(lambda: dict(gt=0, pred=0, hit=0, err=[]))
    for _ in range(steps):
        pred, gt = unravel(np.asarray(obs[-1])), unravel(np.asarray(gt_flat(st.atari_state.env_state)))
        for name, f in env.fields.items():
            pv, gv = getattr(pred, name), getattr(gt, name)
            if f["kind"] == "oo":
                g = _visible(gv, screen_w, screen_h)
                a = np.atleast_1d(np.asarray(pv.active)).ravel() == 1
                p = np.stack([np.atleast_1d(np.asarray(pv.x)).ravel()[a], np.atleast_1d(np.asarray(pv.y)).ravel()[a]], 1)
                errs = _match(g, p, hit_px)
            elif f["kind"] == "xy":
                ga, pa = np.asarray(gv), np.asarray(pv)
                g, p = ga[(ga[:, 0] > 1) & (ga[:, 1] > 1)], pa[pa[:, 0] != MISSING]
                errs = _match(g, p, hit_px)
            elif f["kind"] == "grid":
                e = f["entry"]
                ga = np.asarray(gv, float)
                on = ga >= e["active_min"] if e.get("active_min") is not None else \
                    ga == (e["active_value"] if e.get("active_value") is not None else 1.0)
                pon = np.asarray(pv) != MISSING
                g, p, errs = np.argwhere(on), np.argwhere(pon), [0.0] * int((on & pon).sum())
            else:
                continue
            s = stats[name]
            s["gt"], s["pred"], s["hit"] = s["gt"] + len(g), s["pred"] + len(p), s["hit"] + len(errs)
            s["err"] += errs
        rng, k = jax.random.split(rng)
        obs, st, *_ = step(st, jax.random.randint(k, (), 0, env.action_space().n))

    speed = {}
    for B in batch_sizes:
        vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
        o, s = vreset(jax.random.split(jax.random.PRNGKey(0), B))
        acts = jnp.zeros(B, jnp.int32)
        o, s, *_ = vstep(s, acts)
        jax.block_until_ready(o)
        n = max(3, 256 // B)
        t0 = time.time()
        for _ in range(n):
            o, s, *_ = vstep(s, acts)
        jax.block_until_ready(o)
        speed[B] = round(B * n / (time.time() - t0), 1)

    tot = {k: sum(s[k] for s in stats.values()) for k in ("gt", "pred", "hit")}
    allerr = [e for s in stats.values() for e in s["err"]]
    return dict(
        checkpoint=str(path), game=env.game_name, max_pool=env.max_pool, steps=steps, obs_dim=env.num_features,
        recall=tot["hit"] / max(tot["gt"], 1), precision=tot["hit"] / max(tot["pred"], 1),
        pos_err=float(np.mean(allerr)) if allerr else None, env_steps_per_s=speed,
        fields={n: dict(kind=env.fields[n]["kind"], cls=env.fields[n]["cls"], gt=s["gt"], pred=s["pred"],
                        recall=s["hit"] / max(s["gt"], 1) if s["gt"] else None,
                        precision=s["hit"] / max(s["pred"], 1) if s["pred"] else None,
                        pos_err=float(np.mean(s["err"])) if s["err"] else None)
                for n, s in stats.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 128], help="vmap sizes for the speed test")
    ap.add_argument("--hit-px", type=float, default=3.0)
    ap.add_argument("--max-pool", action="store_true", help="max over the last two frames (flickering games)")
    ap.add_argument("--json", default=None, help="write all results to this file")
    a = ap.parse_args()
    results = []
    print("| checkpoint | recall | precision | pos. err (px) | " + " | ".join(f"SPS@{b}" for b in a.batch) + " |")
    print("|---|---|---|---|" + "---|" * len(a.batch))
    for c in a.checkpoints:
        r = evaluate(c, a.steps, a.batch, a.hit_px, a.max_pool)
        results.append(r)
        err = "–" if r["pos_err"] is None else f"{r['pos_err']:.2f}"
        print(f"| {Path(c).parent.name}/{Path(c).stem} | {r['recall']:.1%} | {r['precision']:.1%} | "
              f"{err} | " + " | ".join(str(r["env_steps_per_s"][b]) for b in a.batch) + " |", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
