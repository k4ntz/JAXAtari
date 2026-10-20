"""Record one episode with YOLOObjectCentricWrapper and overlay the detections on the game frames.

    python train_detection/render_episode.py gray96/pong                     # random policy
    python train_detection/render_episode.py rgb640/seaquest --policy down
    python train_detection/render_episode.py rgb640/beamrider --policy fire --out beamrider.mp4

--policy is "random" or a JaxtariAction name held for the whole episode (fire, down, upfire, ...).
By default boxes are drawn on the full-resolution colour frame; --detector-view shows the frame exactly as
the detector sees it instead (e.g. 96x96 grayscale from native downscaling), upscaled for viewing; --raw
draws the detections (1 px boxes, no labels) on the detector frames at their native size and also saves the
untouched frames and the detections as <out>.npz.
The episode runs until game over (episodic_life=False). Requires opencv + imageio.
"""
import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

import jaxtari
from jaxtari.detection import YOLOObjectCentricWrapper, load_detector
from jaxtari.environment import JaxtariAction as Action
from jaxtari.wrappers import AtariWrapper

COLORS = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 215, 0), (170, 55, 219), (180, 180, 180), (255, 255, 255)]
SCALE = 4  # game frame upscale for the video (160x210 -> 640x840)


def draw(frame, det, names, box_scale, scale=SCALE):
    """frame: (H, W, 1|3) uint8; box_scale: detector frame px -> frame px."""
    frame = np.asarray(frame)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, -1)
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    v = np.asarray(det.valid)
    for (x1, y1, x2, y2), c, s in zip(np.asarray(det.boxes)[v] * box_scale * scale, np.asarray(det.cls)[v],
                                      np.asarray(det.conf)[v]):
        col = COLORS[int(c) % len(COLORS)]
        label = f"{names.get(int(c), int(c))} {float(s):.2f}"
        if scale == 1:  # native size: 1 px boxes on the pixel grid, no labels
            cv2.rectangle(img, (int(np.floor(x1)), int(np.floor(y1))), (int(np.ceil(x2)) - 1, int(np.ceil(y2)) - 1), col, 1)
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), col, -1)
        cv2.putText(img, label, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="checkpoint name (e.g. gray96/pong) or .npz path")
    ap.add_argument("--policy", default="random")
    ap.add_argument("--out", default=None, help="video path (default: <game>_<policy>.mp4)")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detector-view", action="store_true", help="show the frame as the detector sees it")
    ap.add_argument("--raw", action="store_true", help="detector frames at native size, 1 px boxes (+ .npz)")
    a = ap.parse_args()

    det = load_detector(a.checkpoint)
    env = YOLOObjectCentricWrapper(AtariWrapper(jaxtari.make(det.game), episodic_life=False), det)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    if a.raw:
        a.detector_view = True
    if a.detector_view:  # the (possibly natively downscaled, grayscale) frame the detector gets
        render, box_scale = jax.jit(env.render_frame), 1.0
        scale = 1 if a.raw else max(1, round(840 / det.frame_size[1]))
    else:  # full-resolution colour frames, boxes mapped to game pixels
        full = jaxtari.make(det.game)
        render_full = jax.jit(full.render)
        render, box_scale, scale = (lambda s: render_full(s.env_state)), 1.0 / env.scale, SCALE
    n_act = env.action_space().n
    fixed = None
    if a.policy != "random":
        base = env._env._env
        action_set = np.asarray(getattr(base, "ACTION_SET", getattr(base, "action_set", np.arange(n_act)))).tolist()
        fixed = jnp.int32(action_set.index(int(getattr(Action, a.policy.upper()))))

    suffix = "_raw" if a.raw else "_detector_view" if a.detector_view else ""
    out = Path(a.out or f"{det.game}_{a.policy}{suffix}.mp4")
    raw = dict(frames=[], boxes=[], conf=[], cls=[], valid=[])

    def frame(state):
        f = render(state.atari_state)
        if a.raw:  # keep the untouched detector input and detections
            for k, v in (("frames", f), ("boxes", state.detections.boxes), ("conf", state.detections.conf),
                         ("cls", state.detections.cls), ("valid", state.detections.valid)):
                raw[k].append(np.asarray(v))
        return draw(f, state.detections, det.names, box_scale, scale)

    rng = jax.random.PRNGKey(a.seed)
    rng, k = jax.random.split(rng)
    obs, st = reset(k)
    writer = imageio.get_writer(out, fps=a.fps, macro_block_size=8 if not a.raw else 16,
                                ffmpeg_params=["-crf", "0"] if a.raw else None)  # raw: lossless-ish encoding
    writer.append_data(frame(st))
    n_det, ret, t0 = [], 0.0, time.time()
    for t in range(a.max_steps):
        rng, k = jax.random.split(rng)
        action = fixed if fixed is not None else jax.random.randint(k, (), 0, n_act)
        obs, st, reward, terminated, truncated, info = step(st, action)
        if bool(info["env_done"]) or bool(truncated):
            break  # the wrapper auto-resets: stop before the next episode's first frame
        ret += float(info.get("env_reward", reward))
        writer.append_data(frame(st))
        n_det.append(int(np.asarray(st.detections.valid).sum()))
    writer.close()
    if a.raw:  # exact detector frames + detections (detector frame pixels)
        np.savez_compressed(out.with_suffix(".npz"), names=np.asarray([det.names[i] for i in sorted(det.names)]),
                            **{k: np.stack(v) for k, v in raw.items()})
    print(f"{det.game}: {t + 1} agent steps, return {ret:g}, {np.mean(n_det):.2f} detections/frame, "
          f"{time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
