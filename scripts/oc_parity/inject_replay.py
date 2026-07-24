#!/usr/bin/env python3
"""Inject an OCAtari trajectory frame into JAXAtari Pong and optionally keep playing.

Examples:
  python scripts/oc_parity/inject_replay.py --traj /tmp/pong.npz --frame auto --compare /tmp/cmp.png
  python scripts/oc_parity/inject_replay.py --traj /tmp/pong.npz --play
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

# Force JAX on CPU before importing jax when requested.
if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from oc_parity.trajectory_io import (  # noqa: E402
    find_first_ball_frame,
    load_trajectory,
    summarize_trajectory,
)
from oc_parity.translators.pong import trajectory_frame_to_pong_state  # noqa: E402


def _map_action_to_index(env, action_const: int) -> jnp.ndarray:
    action_set = np.asarray(env.ACTION_SET)
    matches = np.where(action_set == int(action_const))[0]
    if len(matches) == 0:
        noop = np.where(action_set == 0)[0]
        idx = int(noop[0]) if len(noop) else 0
    else:
        idx = int(matches[0])
    return jnp.asarray(idx, dtype=jnp.int32)


def _side_by_side(oc_frame: np.ndarray, jax_frame: np.ndarray) -> np.ndarray:
    oc = np.asarray(oc_frame, dtype=np.uint8)
    jf = np.asarray(jax_frame, dtype=np.uint8)
    if oc.shape != jf.shape:
        # Pad / crop to common H,W
        h = min(oc.shape[0], jf.shape[0])
        w = min(oc.shape[1], jf.shape[1])
        oc = oc[:h, :w]
        jf = jf[:h, :w]
    return np.concatenate([oc, jf], axis=1)


def _save_compare_png(path: str, oc_frame: Optional[np.ndarray], jax_frame: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:
        # Fallback without PIL: write raw via imageio or skip
        try:
            import imageio.v2 as imageio
        except ImportError:
            np.save(path + ".npy", jax_frame)
            print(f"PIL/imageio unavailable; wrote JAX frame npy next to {path}")
            return
        if oc_frame is None:
            imageio.imwrite(path, np.asarray(jax_frame, dtype=np.uint8))
        else:
            imageio.imwrite(path, _side_by_side(oc_frame, jax_frame))
        print(f"Wrote comparison image: {path}")
        return

    if oc_frame is None:
        Image.fromarray(np.asarray(jax_frame, dtype=np.uint8)).save(path)
    else:
        Image.fromarray(_side_by_side(oc_frame, jax_frame)).save(path)
    print(f"Wrote comparison image: {path}")


def _interactive_play(env, state, *, fps: int = 30) -> None:
    import pygame
    from utils import get_human_action, update_pygame

    pygame.init()
    scale = 4
    h, w = 210, 160
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("JAXAtari inject-replay — ESC quit, P pause")
    clock = pygame.time.Clock()

    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    paused = False
    running = True
    while running:
        # Drain events for pause / quit (get_human_action also polls QUIT)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused

        if not running:
            break

        if not paused:
            action_const = int(np.asarray(get_human_action()))
            action_idx = _map_action_to_index(env, action_const)
            _obs, state, _r, done, _info = jitted_step(state, action_idx)
            if bool(done):
                print("Episode done — holding final state (press R not wired; ESC to quit)")

        raster = jitted_render(state)
        update_pygame(screen, raster, scale, w, h)
        clock.tick(fps)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject OCAtari trajectory frame into JAXAtari Pong."
    )
    parser.add_argument("--traj", type=str, required=True, help="Path to OC trajectory .npz")
    parser.add_argument(
        "--frame",
        type=str,
        default="auto",
        help="Frame index, or 'auto' for first live Ball frame",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override traj seed for reset key")
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Optional path to write OC|JAX side-by-side PNG",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="After inject, open interactive JAX play from that state",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cpu", action="store_true", help="Force JAX CPU")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    import jaxatari

    traj = load_trajectory(args.traj)
    if args.summarize:
        print(summarize_trajectory(traj))

    if args.frame == "auto":
        t = find_first_ball_frame(traj)
        if t is None:
            t = 0
            print("Warning: no Ball frame found; using t=0")
        else:
            print(f"Auto-selected first Ball frame t={t}")
    else:
        t = int(args.frame)

    n = len(traj["actions"])
    if t < 0 or t >= n:
        raise SystemExit(f"Frame {t} out of range [0, {n})")

    env = jaxatari.make("pong")
    seed = args.seed if args.seed is not None else int(traj["meta"].get("seed", 0))
    state = trajectory_frame_to_pong_state(env, traj, t, seed=seed)

    jitted_render = jax.jit(env.render)
    jax_frame = np.asarray(jitted_render(state), dtype=np.uint8)
    oc_frame = None if traj["frames"] is None else np.asarray(traj["frames"][t], dtype=np.uint8)

    print(
        "Injected state: "
        f"player_y={float(state.player_y):.1f} player_speed={float(state.player_speed):.2f} "
        f"ball=({int(state.ball_x)},{int(state.ball_y)}) vel=({int(state.ball_vel_x)},{int(state.ball_vel_y)}) "
        f"enemy_y={int(state.enemy_y)} scores=({int(state.player_score)},{int(state.enemy_score)}) "
        f"step_counter={int(state.step_counter)}"
    )

    if args.compare:
        out_dir = os.path.dirname(os.path.abspath(args.compare))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _save_compare_png(args.compare, oc_frame, jax_frame)

    if args.play:
        _interactive_play(env, state, fps=args.fps)


if __name__ == "__main__":
    main()
