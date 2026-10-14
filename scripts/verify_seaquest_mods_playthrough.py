#!/usr/bin/env python3
"""Manual playthrough for Seaquest mods that recently changed.

Cycles only the re-verify set (bugfixes / behavior tweaks) one at a time.

Controls
--------
  Arrows / WASD + Space  play
  N                       next mod (rebuilds env, clears JIT caches)
  B                       previous mod
  R                       reset current mod
  P                       pause
  ESC / close window      quit

The active mod name is printed to the terminal and shown in the window title
and as an on-screen overlay.
"""

from __future__ import annotations

import os
import sys

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pygame

from jaxtari.core import make as jaxtari_make
from utils import get_human_action, update_pygame

UPSCALE_FACTOR = 4

# Only mods that changed since the last playthrough (bugfixes / behavior tweaks).
VERIFY_MODS = [
    "faster_enemies",
    "surface_sub_always",
    "swap_diver_shark_roles",
    "no_surfacing_death",
    "inverted_controls",
    "tiny_divers",
]


def _map_action_to_index(env, action_constant):
    if hasattr(env, "ACTION_SET"):
        action_set = np.array(env.ACTION_SET)
        matches = np.where(action_set == int(action_constant))[0]
        if len(matches) > 0:
            return jnp.array(int(matches[0]), dtype=jnp.int32)
        return jnp.array(0, dtype=jnp.int32)
    return jnp.array(action_constant, dtype=jnp.int32)


def _overlay_mod_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Bake a simple text banner into the RGB frame via pygame (CPU, once/frame)."""
    h, w = frame.shape[:2]
    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2).copy())
    font = pygame.font.SysFont("dejavusans", max(10, h // 14), bold=True)
    text = font.render(label, True, (255, 255, 0))
    bar = pygame.Surface((w, text.get_height() + 4))
    bar.set_alpha(160)
    bar.fill((0, 0, 0))
    surf.blit(bar, (0, 0))
    surf.blit(text, (4, 2))
    return pygame.surfarray.array3d(surf).transpose(1, 0, 2)


def _build_env(mod_key: str, seed: int):
    jax.clear_caches()
    env = jaxtari_make(game_name="seaquest", mods=[mod_key])
    key = jrandom.PRNGKey(seed)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)
    obs, state = jitted_reset(key)
    return env, jitted_reset, jitted_step, jitted_render, obs, state, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Force JAX on CPU.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start at this mod key (skip earlier ones).",
    )
    args = parser.parse_args()

    mods = list(VERIFY_MODS)
    idx = 0
    if args.start:
        if args.start not in mods:
            print(f"Unknown --start {args.start!r}. Choices:\n  " + "\n  ".join(mods))
            sys.exit(1)
        idx = mods.index(args.start)

    pygame.init()
    pygame.font.init()

    def load(i: int):
        mod = mods[i]
        print("\n" + "=" * 60)
        print(f"[{i + 1}/{len(mods)}] CURRENT MOD: {mod}")
        print("=" * 60)
        print("Keys: N=next  B=prev  R=reset  P=pause  ESC=quit")
        env, j_reset, j_step, j_render, obs, state, key = _build_env(mod, args.seed + i)
        shape = j_render(state).shape[:2]
        window = pygame.display.set_mode(
            (shape[1] * UPSCALE_FACTOR, shape[0] * UPSCALE_FACTOR)
        )
        pygame.display.set_caption(
            f"Seaquest mod verify [{i + 1}/{len(mods)}]: {mod}"
        )
        return env, j_reset, j_step, j_render, obs, state, key, window, mod

    env, j_reset, j_step, j_render, obs, state, key, window, mod = load(idx)
    clock = pygame.time.Clock()
    pause = False
    running = True
    total_return = 0.0
    reset_counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    pause = not pause
                elif event.key == pygame.K_r:
                    key = jrandom.fold_in(jrandom.PRNGKey(args.seed + idx), reset_counter)
                    reset_counter += 1
                    obs, state = j_reset(key)
                    total_return = 0.0
                    print(f"  reset  | mod={mod}")
                elif event.key == pygame.K_n:
                    if idx + 1 >= len(mods):
                        print("Already on last mod.")
                    else:
                        idx += 1
                        (
                            env,
                            j_reset,
                            j_step,
                            j_render,
                            obs,
                            state,
                            key,
                            window,
                            mod,
                        ) = load(idx)
                        total_return = 0.0
                        reset_counter = 1
                elif event.key == pygame.K_b:
                    if idx <= 0:
                        print("Already on first mod.")
                    else:
                        idx -= 1
                        (
                            env,
                            j_reset,
                            j_step,
                            j_render,
                            obs,
                            state,
                            key,
                            window,
                            mod,
                        ) = load(idx)
                        total_return = 0.0
                        reset_counter = 1

        if not pause:
            # Movement uses get_pressed(); KEYDOWN for N/B/R already handled above.
            # Drain leftover events so get_human_action does not re-handle QUIT oddly.
            action = _map_action_to_index(env, get_human_action())
            obs, state, reward, done, info = j_step(state, action)
            total_return += float(reward)
            if done:
                print(f"  episode done  return={total_return:.0f}  | mod={mod}")
                key = jrandom.fold_in(jrandom.PRNGKey(args.seed + idx), reset_counter)
                reset_counter += 1
                obs, state = j_reset(key)
                total_return = 0.0

        frame = np.asarray(j_render(state), dtype=np.uint8)
        frame = _overlay_mod_label(frame, f"[{idx + 1}/{len(mods)}] {mod}")
        update_pygame(window, frame, UPSCALE_FACTOR, 160, 210)
        clock.tick(args.fps)

    pygame.quit()
    print(f"\nStopped on [{idx + 1}/{len(mods)}] {mod}")


if __name__ == "__main__":
    main()
