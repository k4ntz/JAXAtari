#!/usr/bin/env python3
"""Record OCAtari trajectories (objects + RAM + frames + actions).

Examples:
  python scripts/oc_parity/record_ocatari.py -g Pong --mode play --skip-frames 500 --steps 800 --out /tmp/pong.npz
  python scripts/oc_parity/record_ocatari.py -g Pong --mode random --skip-frames 500 --steps 800 --out /tmp/pong.npz

Requires: pip install ocatari pygame
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

# Configure SDL early when play mode is requested (before OCAtari/pygame import).
if "--mode" in sys.argv:
    try:
        _mode_idx = sys.argv.index("--mode")
        if _mode_idx + 1 < len(sys.argv) and sys.argv[_mode_idx + 1] == "play":
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")
            os.environ.setdefault("SDL_RENDER_DRIVER", "software")
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
            os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    except ValueError:
        pass

import numpy as np

# Allow `python scripts/oc_parity/record_ocatari.py` imports.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from oc_parity.trajectory_io import (  # noqa: E402
    save_trajectory,
    serialize_oc_objects,
    summarize_trajectory,
    load_trajectory,
)


def _configure_pygame_display() -> None:
    """Prefer software/Mesa GL to avoid GLX BadValue crashes on some NVIDIA/X11 setups.

    Must run before ``import pygame`` / OCAtari (which may import pygame).
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    # Critical on this machine: NVIDIA GLX CreateContext returns BadValue.
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
    # Quiet noisy audio init when we only need a window.
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _require_ocatari():
    try:
        from ocatari.core import OCAtari
    except ImportError as exc:
        raise SystemExit(
            "OCAtari is required. Install with: pip install ocatari==2.2.1"
        ) from exc
    return OCAtari


def make_oc_env(game: str, seed: int, *, render_mode: Optional[str] = "rgb_array"):
    OCAtari = _require_ocatari()
    env = OCAtari(
        game,
        mode="ram",
        hud=True,
        frameskip=1,
        obs_mode="obj",
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


def _action_meanings(env) -> List[str]:
    try:
        return list(env.get_action_meanings())
    except Exception:
        try:
            return list(env.unwrapped.get_action_meanings())
        except Exception:
            n = int(env.action_space.n)
            return [f"ACTION_{i}" for i in range(n)]


def _human_action_from_keys(meanings: Sequence[str], pressed) -> int:
    """Map pressed keys to an OCAtari action index using action meanings."""
    import pygame

    meaning_to_idx = {m.upper(): i for i, m in enumerate(meanings)}
    noop = meaning_to_idx.get("NOOP", 0)

    up = pressed[pygame.K_UP] or pressed[pygame.K_w]
    down = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
    left = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
    fire = pressed[pygame.K_SPACE] or pressed[pygame.K_RETURN]

    candidates = []
    if up and right and fire:
        candidates.append("UPRIGHTFIRE")
    if up and left and fire:
        candidates.append("UPLEFTFIRE")
    if down and right and fire:
        candidates.append("DOWNRIGHTFIRE")
    if down and left and fire:
        candidates.append("DOWNLEFTFIRE")
    if up and fire:
        candidates.append("UPFIRE")
    if down and fire:
        candidates.append("DOWNFIRE")
    if left and fire:
        candidates.append("LEFTFIRE")
    if right and fire:
        candidates.append("RIGHTFIRE")
    if up and right:
        candidates.append("UPRIGHT")
    if up and left:
        candidates.append("UPLEFT")
    if down and right:
        candidates.append("DOWNRIGHT")
    if down and left:
        candidates.append("DOWNLEFT")
    if fire:
        candidates.append("FIRE")
    if up:
        candidates.append("UP")
    if down:
        candidates.append("DOWN")
    if left:
        candidates.append("LEFT")
    if right:
        candidates.append("RIGHT")

    for name in candidates:
        if name in meaning_to_idx:
            return meaning_to_idx[name]
    return noop


def _get_rgb(env) -> np.ndarray:
    frame = env.render()
    if frame is None:
        # Fallback: some OCAtari builds need ale.getScreenRGB
        try:
            frame = env._env.unwrapped.ale.getScreenRGB()
        except Exception:
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
    return np.asarray(frame, dtype=np.uint8)


def _snapshot_step(env, action: int, store_frames: bool) -> Tuple[
    int, float, bool, bool, np.ndarray, List[dict], Optional[np.ndarray]
]:
    obs, reward, terminated, truncated, info = env.step(int(action))
    del obs, info
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    objects = serialize_oc_objects(env.objects)
    frame = _get_rgb(env) if store_frames else None
    return (
        int(action),
        float(reward),
        bool(terminated),
        bool(truncated),
        ram,
        objects,
        frame,
    )


def record_random(
    env,
    *,
    steps: int,
    seed: int,
    store_frames: bool,
    skip_frames: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n_actions = int(env.action_space.n)

    actions: List[int] = []
    rewards: List[float] = []
    terminateds: List[bool] = []
    truncateds: List[bool] = []
    rams: List[np.ndarray] = []
    objects: List[List[dict]] = []
    frames: Optional[List[np.ndarray]] = [] if store_frames else None

    done = False
    total = int(skip_frames) + int(steps)
    for t in range(total):
        if done:
            env.reset(seed=seed + t + 1)
            done = False
        action = int(rng.integers(0, n_actions))
        a, r, term, trunc, ram, objs, frame = _snapshot_step(env, action, store_frames)
        done = term or trunc
        # Warmup: step but do not save (avoids spawn / serve artifacts in the traj).
        if t < int(skip_frames):
            continue
        actions.append(a)
        rewards.append(r)
        terminateds.append(term)
        truncateds.append(trunc)
        rams.append(ram)
        objects.append(objs)
        if frames is not None and frame is not None:
            frames.append(frame)

    return {
        "actions": actions,
        "rewards": rewards,
        "terminated": terminateds,
        "truncated": truncateds,
        "ram": rams,
        "objects": objects,
        "frames": frames,
    }


def record_play(
    env,
    *,
    steps: Optional[int],
    store_frames: bool,
    fps: int,
    skip_frames: int = 0,
) -> dict:
    import pygame

    meanings = _action_meanings(env)
    pygame.display.init()
    pygame.font.init()
    scale = 3
    native_h, native_w = 210, 160
    try:
        screen = pygame.display.set_mode((native_w * scale, native_h * scale))
    except pygame.error as exc:
        raise SystemExit(
            "Failed to open a pygame window for play mode.\n"
            f"  {exc}\n"
            "Try:  LIBGL_ALWAYS_SOFTWARE=1 SDL_RENDER_DRIVER=software \\\n"
            "      python scripts/oc_parity/record_ocatari.py -g Pong --mode play ...\n"
            "Or record with --mode random if you have no working display."
        ) from exc
    pygame.display.set_caption(
        "OCAtari recorder — ESC quit, P pause "
        f"(warmup {skip_frames} frames not saved)"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    actions: List[int] = []
    rewards: List[float] = []
    terminateds: List[bool] = []
    truncateds: List[bool] = []
    rams: List[np.ndarray] = []
    objects: List[List[dict]] = []
    frames: Optional[List[np.ndarray]] = [] if store_frames else None

    paused = False
    running = True
    done = False
    warmup_left = int(max(0, skip_frames))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused

        if steps is not None and len(actions) >= steps:
            break

        pressed = pygame.key.get_pressed()
        if not paused:
            if done:
                env.reset()
                done = False
            action = _human_action_from_keys(meanings, pressed)
            a, r, term, trunc, ram, objs, frame = _snapshot_step(
                env, action, store_frames=store_frames
            )
            done = term or trunc
            if warmup_left > 0:
                warmup_left -= 1
            else:
                actions.append(a)
                rewards.append(r)
                terminateds.append(term)
                truncateds.append(trunc)
                rams.append(ram)
                objects.append(objs)
                if frames is not None and frame is not None:
                    frames.append(frame)
            display_frame = frame if frame is not None else _get_rgb(env)
        else:
            display_frame = _get_rgb(env)

        if display_frame is not None:
            surf = pygame.surfarray.make_surface(
                np.transpose(display_frame, (1, 0, 2))
            )
            surf = pygame.transform.scale(surf, (native_w * scale, native_h * scale))
            screen.blit(surf, (0, 0))
            if warmup_left > 0:
                label = f"WARMUP {skip_frames - warmup_left}/{skip_frames} (not saved)"
            else:
                label = f"RECORDING frames={len(actions)} paused={paused}"
            screen.blit(font.render(label, True, (255, 255, 0)), (8, 8))
            pygame.display.flip()

        clock.tick(fps)

    pygame.quit()
    if not store_frames:
        frames = None
    return {
        "actions": actions,
        "rewards": rewards,
        "terminated": terminateds,
        "truncated": truncateds,
        "ram": rams,
        "objects": objects,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record OCAtari trajectories for JAX parity.")
    parser.add_argument("-g", "--game", type=str, default="Pong")
    parser.add_argument(
        "--mode",
        choices=("random", "play"),
        default="random",
        help="random: scripted random actions; play: interactive pygame",
    )
    parser.add_argument("--steps", type=int, default=240, help="Frames to record after warmup (random; optional cap for play)")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=500,
        help="Warmup frames to step but NOT save (default 500; avoids spawn/serve). Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--no-frames", action="store_true", help="Do not store RGB frames")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS in play mode")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="After saving, reload and print a short summary",
    )
    args = parser.parse_args()

    # Play mode needs a window; configure SDL *before* OCAtari imports pygame.
    if args.mode == "play":
        _configure_pygame_display()

    store_frames = not args.no_frames
    render_mode = "rgb_array"
    env = make_oc_env(args.game, args.seed, render_mode=render_mode)
    meanings = _action_meanings(env)

    if args.skip_frames:
        print(f"Warmup: skipping first {args.skip_frames} frames (not saved)")

    if args.mode == "random":
        data = record_random(
            env,
            steps=args.steps,
            seed=args.seed,
            store_frames=store_frames,
            skip_frames=args.skip_frames,
        )
    else:
        # In play mode, --steps is an optional cap (None = until quit) if user
        # passes a negative value; default still caps at --steps.
        steps_cap = args.steps if args.steps > 0 else None
        data = record_play(
            env,
            steps=steps_cap,
            store_frames=store_frames,
            fps=args.fps,
            skip_frames=args.skip_frames,
        )

    env.close()

    meta = {
        "game": args.game,
        "seed": args.seed,
        "frameskip": 1,
        "oc_mode": "ram",
        "hud": True,
        "obs_mode": "obj",
        "action_meanings": meanings,
        "record_mode": args.mode,
        "skip_frames": int(args.skip_frames),
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_trajectory(
        args.out,
        meta=meta,
        actions=data["actions"],
        rewards=data["rewards"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        ram=data["ram"],
        objects=data["objects"],
        frames=data["frames"],
    )
    print(f"Saved {len(data['actions'])} frames → {args.out}")

    if args.summarize:
        traj = load_trajectory(args.out)
        print(summarize_trajectory(traj))


if __name__ == "__main__":
    main()
