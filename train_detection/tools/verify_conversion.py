"""Check that a converted .npz detector reproduces ultralytics' own predictions (needs torch/ultralytics).

    python train_detection/tools/verify_conversion.py path/to/best.pt pong_gray96.npz

Runs both on frames from random play (rendered exactly as the wrapper renders them) and reports the
fraction of ultralytics detections reproduced (same class, IoU > 0.5) and the largest box difference.
"""
import argparse

import cv2
import jax
import jax.numpy as jnp
import numpy as np

import jaxtari
from jaxtari.detection import load_detector


def frames(det, n=30, seed=0):
    env = jaxtari.make(det.game)
    if det.input_mode == "native":
        from jaxtari.modification import apply_native_downscaling
        apply_native_downscaling(env, (det.frame_size[1], det.frame_size[0]),
                                 int(det.params[0]["w"].shape[2]) == 1)
    reset, step, render = jax.jit(env.reset), jax.jit(env.step), jax.jit(env.render)
    key = jax.random.PRNGKey(seed)
    _, st = reset(key)
    out = []
    for t in range(n * 7):
        key, k = jax.random.split(key)
        _, st, _, done, _ = step(st, jax.random.randint(k, (), 0, env.action_space().n))
        if bool(done):
            _, st = reset(k)
        if t % 7 == 3:
            out.append(np.asarray(render(st)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt")
    ap.add_argument("npz")
    ap.add_argument("--frames", type=int, default=30)
    a = ap.parse_args()
    from ultralytics import YOLO
    yt = YOLO(a.pt)
    det = load_detector(a.npz)
    detect = jax.jit(det.detect)
    imgsz = max(det.net_shape) if det.input_mode == "native" else 640 if det.frame_size == (640, 840) else None
    n_ref = n_hit = 0
    worst = 0.0
    for f in frames(det, a.frames):
        img = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), det.frame_size, interpolation=cv2.INTER_NEAREST) \
            if det.input_mode == "rgb" else f
        r = yt.predict(img, imgsz=imgsz or det.net_shape[0], conf=det.conf, iou=det.iou, device="cpu", verbose=False)[0].boxes
        ref = np.concatenate([r.xyxy.numpy(), r.cls.numpy()[:, None]], 1)
        d = detect(det.params, jnp.asarray(f))
        v = np.asarray(d.valid)
        mine = np.concatenate([np.asarray(d.boxes)[v], np.asarray(d.cls)[v][:, None]], 1)
        used = set()
        for b in ref:
            for j, m in enumerate(mine):
                if j in used or m[4] != b[4]:
                    continue
                ix = max(0, min(b[2], m[2]) - max(b[0], m[0]))
                iy = max(0, min(b[3], m[3]) - max(b[1], m[1]))
                u = (b[2] - b[0]) * (b[3] - b[1]) + (m[2] - m[0]) * (m[3] - m[1]) - ix * iy
                if u > 0 and ix * iy / u > 0.5:
                    used.add(j)
                    worst = max(worst, float(np.abs(b[:4] - m[:4]).max()))
                    break
        n_ref, n_hit = n_ref + len(ref), n_hit + len(used)
    print(f"{det.game}: reproduced {n_hit}/{n_ref} ultralytics detections ({n_hit / max(n_ref, 1):.2%}), "
          f"max box difference {worst:.3f} px")


if __name__ == "__main__":
    main()
