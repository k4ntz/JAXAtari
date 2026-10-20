"""Generate YOLO training data from the game state (labels from the quirks registry).

Frames are sampled from random play; labels follow jaxtari.detection.labels (registry geometry, 8 px
floor) plus the OpenCV fallbacks in game_patches.py for objects the game state does not expose.

    # 96x96 grayscale frames from JAXtari native downscaling (what the gray96 detectors use)
    python train_detection/generate_dataset.py pong asteroids --size 96 96
    # class balancing for games where the player is rarely alive under random play
    python train_detection/generate_dataset.py asteroids --size 96 96 --frames 3000 --player-frac 0.4
    # full-resolution RGB frames, nearest-upscaled to 640x840 (what the rgb640 detectors use)
    python train_detection/generate_dataset.py pong --rgb640
    # check / tune registry geometry: draw the labels on a few frames
    python train_detection/generate_dataset.py pong --preview

Output: <out>/<game>/{images,labels}/{train,val}/*.png|txt and <out>/<game>/data.yaml (ultralytics).
Requires opencv (pip install -e ".[train_detection]").
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import jax
import numpy as np

import jaxtari
from jaxtari.detection import labels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game_patches import GAME_PATCHES  # noqa: E402

ROOT = Path(__file__).resolve().parent
COLORS = [(0, 255, 0), (0, 0, 255), (255, 165, 0), (0, 215, 255), (219, 55, 170), (180, 180, 180), (255, 255, 255)]


def sample(game, n_frames, size=None, rgb640=False, every=20, seed=0, player_frac=0.0,
           reset_every=None):
    """-> list of (image, [yolo label lines]) from random play."""
    objects = labels.load_registry()[game]["objects"]
    env = jaxtari.make(game)
    full = jaxtari.make(game)  # original renderer: native RGB for the OpenCV patches
    H, W = env.renderer.config.game_dimensions
    if size is not None:
        from jaxtari.modification import apply_native_downscaling
        apply_native_downscaling(env, (size[1], size[0]), True)
    reset, step, render, render_full = jax.jit(env.reset), jax.jit(env.step), jax.jit(env.render), \
        jax.jit(full.render)
    patch = GAME_PATCHES.get(game)
    n_act = env.action_space().n
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    obs, st = reset(key)
    samples, t, next_t, since_reset = [], 0, every, 0
    while len(samples) < n_frames:
        a = 1 if rng.random() < 0.5 else int(rng.integers(n_act))  # 50% FIRE, as the original extractor
        obs, st, _, done, _ = step(st, a)
        t, since_reset = t + 1, since_reset + 1
        if bool(done) or (reset_every and since_reset >= reset_every):
            key, k = jax.random.split(key)
            obs, st = reset(k)
            since_reset = 0
        if t < next_t:
            continue
        next_t = t + every + int(rng.integers(0, 4))  # jitter: sample both parities of flickering games
        lines = labels.frame_labels(obs, objects, W, H)
        if patch is not None:
            pixels = np.asarray(render_full(st))
            patch(pixels, lines, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                  lambda c, x, y, w, h, ah, aw: labels.to_yolo_line(c, x, y, w, h, aw, ah),
                  lambda *args: None, False)
        if not lines:
            continue  # frames without objects are skipped
        if player_frac > 0 and not any(l.startswith("0 ") for l in lines):
            n_player = sum(any(l.startswith("0 ") for l in ls) for _, ls in samples)
            if len(samples) - n_player >= n_frames * (1 - player_frac):
                continue  # class balancing for games where the player is rarely alive under random play
        frame = np.asarray(render(st))
        if rgb640:  # native RGB -> 4x nearest upscale (640x840), BGR for cv2
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (640, 840), interpolation=cv2.INTER_NEAREST)
        elif frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        samples.append((frame, lines))
    return samples


def write(samples, out, val_frac=0.2):
    shutil.rmtree(out, ignore_errors=True)
    order = list(range(len(samples)))
    random.Random(42).shuffle(order)
    n_val = int(len(samples) * val_frac)
    for split, idx in (("val", order[:n_val]), ("train", order[n_val:])):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for i in idx:
            img, lines = samples[i]
            cv2.imwrite(str(out / "images" / split / f"frame_{i:05d}.png"), img)
            (out / "labels" / split / f"frame_{i:05d}.txt").write_text("\n".join(lines))
    channels = 1 if samples[0][0].ndim == 2 or samples[0][0].shape[-1] == 1 else 3
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                                   f"channels: {channels}\nnc: 7\nnames: {json.dumps(labels.CLASS_NAMES)}\n")
    return len(samples) - n_val, n_val


def preview(samples, path, n=8, width=320):
    """Contact sheet (4 per row) of the first n frames with their label boxes."""
    tiles = []
    for img, lines in samples[:n]:
        im = img if img.ndim == 3 and img.shape[-1] == 3 else cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2BGR)
        h = round(im.shape[0] * width / im.shape[1])
        im = cv2.resize(im, (width, h), interpolation=cv2.INTER_NEAREST)
        for l in lines:
            c, x, y, bw, bh = map(float, l.split())
            cv2.rectangle(im, (int((x - bw / 2) * width), int((y - bh / 2) * h)),
                          (int((x + bw / 2) * width), int((y + bh / 2) * h)), COLORS[int(c) % len(COLORS)], 1)
        tiles.append(im)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    cv2.imwrite(str(path), np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--frames", type=int, default=1000)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), help="native downscaled grayscale frames")
    mode.add_argument("--rgb640", action="store_true", help="native RGB, nearest-upscaled to 640x840")
    ap.add_argument("--player-frac", type=float, default=0.0, help="min. fraction of frames containing a player")
    ap.add_argument("--every", type=int, default=20, help="sample a frame every ~N env steps")
    ap.add_argument("--reset-every", type=int, default=None, help="force a reset every N steps (e.g. hauntedhouse)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="only write a label preview image (registry tuning)")
    a = ap.parse_args()
    for g in a.games:
        n = 8 if a.preview else a.frames
        samples = sample(g, n, a.size, a.rgb640, a.every, a.seed, a.player_frac, a.reset_every)
        if a.preview:
            print(f"{g}: preview -> {preview(samples, Path(a.out) / f'{g}_preview.png')}")
            continue
        n_train, n_val = write(samples, Path(a.out) / g)
        counts = Counter(l.split()[0] for _, ls in samples for l in ls)
        print(f"{g}: {n_train} train / {n_val} val, boxes per class {dict(sorted(counts.items()))} -> {Path(a.out) / g}")


if __name__ == "__main__":
    main()
