#!/usr/bin/env python3
"""
Record a human descent of the REAL H.E.R.O. ROM (via ALE) and stitch the
vertically-scrolling shaft into one tall ground-truth map of a level.

This is the reliable way to capture H.E.R.O.'s multi-chamber level geometry:
auto-navigation can't thread the narrow gaps, but a human can. As you play,
every frame's cave region is captured and, on exit, stitched together by
detecting vertical scroll between consecutive frames. The result is a single
tall PNG of the whole shaft you descended, plus the raw frames, which we use
to author the exact world geometry in jax_hero.py.

Controls:  Arrows / WASD = move,  Space = fire,  Down+Space = dynamite,
           P = pause,  R = reset level,  ESC / Q = stop and stitch.

Usage:  python scripts/hero_record_level.py [--out DIR] [--cave-bottom 142]
"""
import os
import argparse
import numpy as np
import pygame
import gymnasium as gym
import ale_py  # noqa: F401  (registers ALE envs)
from PIL import Image

UPSCALE = 3
NATIVE_H, NATIVE_W = 210, 160


def player_semantic(keys):
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE] or keys[pygame.K_RETURN]
    if up and right and fire: return "UPRIGHTFIRE"
    if up and left and fire: return "UPLEFTFIRE"
    if down and fire: return "DOWNFIRE"
    if up and fire: return "UPFIRE"
    if left and fire: return "LEFTFIRE"
    if right and fire: return "RIGHTFIRE"
    if up and right: return "UPRIGHT"
    if up and left: return "UPLEFT"
    if down and right: return "DOWNRIGHT"
    if down and left: return "DOWNLEFT"
    if fire: return "FIRE"
    if up: return "UP"
    if down: return "DOWN"
    if left: return "LEFT"
    if right: return "RIGHT"
    return "NOOP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir for map + frames")
    ap.add_argument("--cave-bottom", type=int, default=142,
                    help="row where the cave ends and the HUD begins (default 142)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "hero_level_capture")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    cave = slice(0, args.cave_bottom)

    env = gym.make("ALE/Hero-v5", frameskip=1, render_mode="rgb_array",
                   repeat_action_probability=0.0, full_action_space=False)
    env.reset(seed=args.seed)
    meanings = env.unwrapped.get_action_meanings()
    A = {n: i for i, n in enumerate(meanings)}

    pygame.init()
    screen = pygame.display.set_mode((NATIVE_W * UPSCALE, NATIVE_H * UPSCALE))
    pygame.display.set_caption("Record H.E.R.O. descent — ESC/Q to stitch & save")
    clock = pygame.time.Clock()

    def show(frame):
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, screen.get_size())
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    strips = []            # stitched world strips, top-first
    prev = None
    frames = []
    running, paused = True, False
    frame = env.render()
    show(frame)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_p:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    env.reset(seed=args.seed)
                    prev = None
                    strips = []
                    frames = []

        if paused:
            clock.tick(30)
            continue

        keys = pygame.key.get_pressed()
        action = A.get(player_semantic(keys), 0)
        _, _, term, trunc, _ = env.step(action)
        frame = env.render()
        frames.append(frame.copy())
        cur = frame[cave].astype(np.int16)

        if prev is None:
            strips.append(cur.copy())
        else:
            best_d, best_err = 0, 1e18
            for d in range(0, 26):
                err = (np.abs(cur - prev).mean() if d == 0
                       else np.abs(cur[:-d] - prev[d:]).mean())
                if err < best_err:
                    best_err, best_d = err, d
            if best_d > 0 and best_err < 12:
                strips.append(cur[-best_d:].copy())
        prev = cur

        show(frame)
        if term or trunc:
            env.reset(seed=args.seed)
            prev = None
        clock.tick(30)

    env.close()
    pygame.quit()

    if strips:
        world = np.concatenate(strips, axis=0).astype(np.uint8)
        Image.fromarray(world).save(os.path.join(out, "level_map.png"))
        print(f"Stitched shaft height: {world.shape[0]}px  ->  {out}/level_map.png")
    # Save a thinned set of raw frames for reference.
    for i in range(0, len(frames), 20):
        Image.fromarray(frames[i]).save(os.path.join(out, f"frame_{i:05d}.png"))
    print(f"Saved {len(range(0, len(frames), 20))} reference frames to {out}")


if __name__ == "__main__":
    main()
