#!/usr/bin/env python3
"""Side-by-side OCAtari | JAXAtari overlay with hotkey state transfer.

Starts with OCAtari only. JAXAtari is constructed on first/later ``T`` press
(nil → live transfer on the right pane).

Controls:
  Arrows / WASD / Space  — actions
  T                     — construct (if needed) + translate OC → JAX (right pane)
  B                     — toggle drive mode: both | jax | oc
  P                     — pause
  F                     — toggle frame-by-frame mode
  N                     — step one frame (when frame-by-frame is on)
  R                     — reset OC; tear down JAX (back to nil)
  ESC / Q               — quit

Requires: pip install ocatari pygame
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

# Force JAX on CPU before importing jax when requested.
if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Soften GLX crashes before any pygame/OCAtari import (overlay always opens a window).
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


UPSCALE = 3
NATIVE_H, NATIVE_W = 210, 160
SCALED_W = NATIVE_W * UPSCALE
SCALED_H = NATIVE_H * UPSCALE
COLOR_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 0)
COLOR_NIL = (40, 40, 40)


def _require_ocatari():
    try:
        from ocatari.core import OCAtari
    except ImportError as exc:
        raise SystemExit(
            "OCAtari is required. Install with: pip install ocatari==2.2.1"
        ) from exc
    return OCAtari


def _action_meanings(env) -> List[str]:
    try:
        return list(env.get_action_meanings())
    except Exception:
        try:
            return list(env.unwrapped.get_action_meanings())
        except Exception:
            return [f"ACTION_{i}" for i in range(int(env.action_space.n))]


def _semantic_from_keys(pressed) -> str:
    import pygame

    up = pressed[pygame.K_UP] or pressed[pygame.K_w]
    down = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
    left = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
    fire = pressed[pygame.K_SPACE] or pressed[pygame.K_RETURN]

    if up and right and fire:
        return "UPRIGHTFIRE"
    if up and left and fire:
        return "UPLEFTFIRE"
    if down and right and fire:
        return "DOWNRIGHTFIRE"
    if down and left and fire:
        return "DOWNLEFTFIRE"
    if up and fire:
        return "UPFIRE"
    if down and fire:
        return "DOWNFIRE"
    if left and fire:
        return "LEFTFIRE"
    if right and fire:
        return "RIGHTFIRE"
    if up and right:
        return "UPRIGHT"
    if up and left:
        return "UPLEFT"
    if down and right:
        return "DOWNRIGHT"
    if down and left:
        return "DOWNLEFT"
    if fire:
        return "FIRE"
    if up:
        return "UP"
    if down:
        return "DOWN"
    if left:
        return "LEFT"
    if right:
        return "RIGHT"
    return "NOOP"


def _oc_action_index(meanings: List[str], semantic: str) -> int:
    mapping = {m.upper(): i for i, m in enumerate(meanings)}
    return int(mapping.get(semantic.upper(), mapping.get("NOOP", 0)))


def _jax_action_set(env) -> np.ndarray:
    if hasattr(env, "ACTION_SET"):
        return np.asarray(env.ACTION_SET)
    if hasattr(env, "action_set"):
        return np.asarray(env.action_set)
    raise AttributeError(f"{type(env).__name__} has no ACTION_SET/action_set")


def _jax_action_index(env, semantic: str):
    import jax.numpy as jnp
    from jaxatari.environment import JAXAtariAction

    const = getattr(JAXAtariAction, semantic, JAXAtariAction.NOOP)
    action_set = _jax_action_set(env)
    matches = np.where(action_set == int(const))[0]
    if len(matches) == 0:
        matches = np.where(action_set == int(JAXAtariAction.NOOP))[0]
        idx = int(matches[0]) if len(matches) else 0
    else:
        idx = int(matches[0])
    return jnp.asarray(idx, dtype=jnp.int32)


def _get_oc_rgb(env) -> np.ndarray:
    frame = env.render()
    if frame is None:
        try:
            frame = env._env.unwrapped.ale.getScreenRGB()
        except Exception:
            frame = np.zeros((NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    return np.asarray(frame, dtype=np.uint8)


def _blit_frame(surface, frame: np.ndarray, x_offset: int) -> None:
    import pygame

    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (SCALED_W, SCALED_H))
    surface.blit(surf, (x_offset, 0))


def _nil_frame() -> np.ndarray:
    frame = np.zeros((NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    frame[:] = COLOR_NIL
    return frame


def _player_xy(objects) -> Optional[Tuple[float, float]]:
    """Return player-like XY (Player, or Freeway Chicken at x≈44)."""
    chickens = []
    for obj in objects:
        cat = (
            getattr(obj, "category", None)
            if not isinstance(obj, dict)
            else obj.get("category")
        )
        x = getattr(obj, "x", None) if not isinstance(obj, dict) else obj.get("x")
        y = getattr(obj, "y", None) if not isinstance(obj, dict) else obj.get("y")
        if cat == "Player" and x is not None and y is not None:
            return float(x), float(y)
        if cat == "Chicken" and x is not None and y is not None:
            chickens.append((float(x), float(y)))
    if chickens:
        # Prefer leftmost (P1 / JAX chicken_x≈44).
        chickens.sort(key=lambda xy: xy[0])
        return chickens[0]
    return None


def _construct_jax(jax_key: str, seed: int) -> dict:
    import jax
    import jaxatari

    print(f"Constructing JAXAtari env '{jax_key}' (first transfer)...")
    env = jaxatari.make(jax_key)
    _obs, _state = env.reset(jax.random.PRNGKey(int(seed)))
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)
    _ = jitted_render(_state)
    print("JAXAtari ready.")
    return {
        "env": env,
        "jitted_step": jitted_step,
        "jitted_render": jitted_render,
    }


def _transfer_summary(jax_key: str, state: Any) -> str:
    if jax_key == "pong":
        return (
            f"player_y={float(state.player_y):.1f} "
            f"ball=({int(state.ball_x)},{int(state.ball_y)}) "
            f"vel=({int(state.ball_vel_x)},{int(state.ball_vel_y)})"
        )
    if jax_key == "bankheist":
        pos = np.asarray(state.player.position)
        return (
            f"player=({int(pos[0])},{int(pos[1])}) dir={int(state.player.direction)} "
            f"fuel={float(state.fuel):.0f} money={int(state.money)} "
            f"lives={int(state.player_lives)} map={int(state.map_id)}"
        )
    return type(state).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="OCAtari | JAXAtari dual-pane overlay")
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        default="Pong",
        help="Game name (OC or JAX key), e.g. Pong, BankHeist, bankheist",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import pygame
    from oc_parity.translators.base import objects_as_dicts
    from oc_parity.translators.registry import (
        get_translator,
        list_implemented_translators,
        normalize_game_name,
        print_disclaimers,
        print_skipped_games_note,
    )

    jax_key, oc_name = normalize_game_name(args.game)
    if jax_key not in list_implemented_translators():
        raise SystemExit(
            f"No translator for '{jax_key}' yet. Implemented: "
            f"{', '.join(list_implemented_translators())}"
        )

    print_skipped_games_note()
    print_disclaimers(jax_key)

    translate = get_translator(jax_key)

    OCAtari = _require_ocatari()
    oc_env = OCAtari(
        oc_name,
        mode="ram",
        hud=True,
        frameskip=1,
        obs_mode="obj",
        render_mode="rgb_array",
    )
    oc_env.reset(seed=args.seed)
    oc_meanings = _action_meanings(oc_env)

    jax_bundle: Optional[dict] = None
    jax_state: Any = None
    frame_index = 0

    # Player (x,y) lookback for direction / speed inference.
    xy_history: Deque[Optional[Tuple[float, float]]] = deque(maxlen=2)
    xy_history.append(_player_xy(oc_env.objects))

    pygame.init()
    screen = pygame.display.set_mode((SCALED_W * 2, SCALED_H))
    pygame.display.set_caption(
        f"OCAtari | JAXAtari [{oc_name}] — T transfer, P pause, F/N frame-step"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    paused = False
    frame_by_frame = False
    next_frame_asked = False
    drive_mode = "both"
    drive_cycle = ["both", "jax", "oc"]
    running = True

    def teardown_jax() -> None:
        nonlocal jax_bundle, jax_state
        jax_bundle = None
        jax_state = None

    def transfer_oc_to_jax() -> None:
        nonlocal jax_bundle, jax_state
        print_disclaimers(jax_key)
        if jax_bundle is None:
            jax_bundle = _construct_jax(jax_key, args.seed)

        objs = objects_as_dicts(oc_env.objects)
        prev_xy = xy_history[-2] if len(xy_history) >= 2 else None

        # Translators accept needed lookback kwargs and ignore the rest.
        kwargs = {
            "seed": args.seed,
            "frame_index": frame_index,
            "print_assumptions": False,
            "prev_player_y": None if prev_xy is None else prev_xy[1],
            "prev_player_xy": prev_xy,
        }

        jax_state = translate(jax_bundle["env"], objs, **kwargs)
        _ = jax_bundle["jitted_render"](jax_state)
        print(
            f"Transferred OC → JAX ({jax_key}) at frame={frame_index} "
            f"{_transfer_summary(jax_key, jax_state)}"
        )

    def step_once(semantic: str) -> None:
        nonlocal jax_state, frame_index
        jax_live = jax_bundle is not None and jax_state is not None
        step_oc = True if not jax_live else drive_mode in ("both", "oc")
        step_jax = drive_mode in ("both", "jax") and jax_live

        if step_oc:
            oc_action = _oc_action_index(oc_meanings, semantic)
            _o, _r, term, trunc, _i = oc_env.step(oc_action)
            if term or trunc:
                oc_env.reset(seed=args.seed + frame_index + 1)
            xy_history.append(_player_xy(oc_env.objects))
            frame_index += 1

        if step_jax:
            jax_action = _jax_action_index(jax_bundle["env"], semantic)
            _o, jax_state, _r, _done, _i = jax_bundle["jitted_step"](
                jax_state, jax_action
            )

    while running:
        next_frame_asked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    print(f"Paused: {paused}")
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    if frame_by_frame:
                        paused = False
                    print(f"Frame-by-frame: {frame_by_frame}")
                elif event.key == pygame.K_n:
                    if frame_by_frame:
                        next_frame_asked = True
                elif event.key == pygame.K_b:
                    drive_mode = drive_cycle[
                        (drive_cycle.index(drive_mode) + 1) % len(drive_cycle)
                    ]
                    print(f"Drive mode: {drive_mode}")
                elif event.key == pygame.K_r:
                    oc_env.reset(seed=args.seed)
                    teardown_jax()
                    frame_index = 0
                    xy_history.clear()
                    xy_history.append(_player_xy(oc_env.objects))
                    print("Reset OC; JAX torn down (nil again)")
                elif event.key == pygame.K_t:
                    transfer_oc_to_jax()

        pressed = pygame.key.get_pressed()
        semantic = _semantic_from_keys(pressed)

        hold = paused or (frame_by_frame and not next_frame_asked)
        if not hold:
            step_once(semantic)

        oc_frame = _get_oc_rgb(oc_env)
        if jax_bundle is not None and jax_state is not None:
            jax_frame = np.asarray(
                jax_bundle["jitted_render"](jax_state), dtype=np.uint8
            )
            jax_label = f"JAXAtari ({jax_key})"
        else:
            jax_frame = _nil_frame()
            jax_label = "JAXAtari (nil — press T)"

        screen.fill(COLOR_BG)
        _blit_frame(screen, oc_frame, 0)
        _blit_frame(screen, jax_frame, SCALED_W)

        if paused:
            mode_tag = "PAUSED"
        elif frame_by_frame:
            mode_tag = "FRAME-BY-FRAME (N)"
        else:
            mode_tag = "LIVE"

        jax_live = jax_bundle is not None and jax_state is not None
        status = (
            f"{oc_name}  {mode_tag}  drive={drive_mode}  "
            f"jax={'live' if jax_live else 'nil'}  frame={frame_index}  "
            f"act={semantic}  [T transfer]"
        )
        screen.blit(font.render(status, True, COLOR_TEXT), (8, 8))
        screen.blit(font.render(f"OCAtari ({oc_name})", True, COLOR_TEXT), (8, SCALED_H - 24))
        screen.blit(
            font.render(jax_label, True, COLOR_TEXT),
            (SCALED_W + 8, SCALED_H - 24),
        )
        pygame.display.flip()
        clock.tick(max(1, args.fps))

    oc_env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
