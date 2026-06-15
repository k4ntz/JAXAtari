"""Quick rendering test harness for jax_icehockey.

Two modes:

  # 1. Headless snapshot — fastest way to eyeball the render while iterating.
  #    Resets the env, renders one frame, writes it to a PNG (upscaled).
  python test.py
  python test.py --out frame.png --scale 6

  # 2. Interactive — open a pygame window and drive the game by hand.
  #    Arrow keys move, Space = FIRE, R = reset, Esc/Q = quit.
  python test.py --play

Run with --cpu to force JAX onto the CPU (handy if a GPU is busy/absent).
"""

import argparse
import os
import sys

# JAX platform must be chosen before `import jax`.
if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.random as jrandom
import numpy as np

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_icehockey import JaxIceHockey

UPSCALE = 4


def to_uint8_image(raster) -> np.ndarray:
    """Convert a (H, W, C) JAX raster into an (H, W, 3) uint8 numpy array."""
    img = np.asarray(raster)
    if img.dtype != np.uint8:
        # Floats are assumed to be in [0, 1].
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    if img.ndim == 2:  # grayscale
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:  # drop alpha
        img = img[..., :3]
    return img


def snapshot(out_path: str, scale: int) -> None:
    env = JaxIceHockey()
    _obs, state = env.reset(jrandom.PRNGKey(0))
    raster = env.render(state)
    img = to_uint8_image(raster)

    print(f"render output: shape={np.asarray(raster).shape} dtype={np.asarray(raster).dtype}")
    print(f"value range: min={img.min()} max={img.max()} (all-black means nothing was drawn)")

    if scale > 1:
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

    try:
        from PIL import Image

        Image.fromarray(img).save(out_path)
        print(f"wrote {out_path} ({img.shape[1]}x{img.shape[0]})")
    except ImportError:
        npy_path = os.path.splitext(out_path)[0] + ".npy"
        np.save(npy_path, img)
        print(f"Pillow not installed; saved raw array to {npy_path} instead.")


KEY_TO_ACTION = {
    # populated inside play() once pygame is imported
}


def get_action(pygame, keys):
    up = keys[pygame.K_UP]
    down = keys[pygame.K_DOWN]
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

    if up and right:
        return Action.UPRIGHTFIRE if fire else Action.UPRIGHT
    if up and left:
        return Action.UPLEFTFIRE if fire else Action.UPLEFT
    if down and right:
        return Action.DOWNRIGHTFIRE if fire else Action.DOWNRIGHT
    if down and left:
        return Action.DOWNLEFTFIRE if fire else Action.DOWNLEFT
    if up:
        return Action.UPFIRE if fire else Action.UP
    if down:
        return Action.DOWNFIRE if fire else Action.DOWN
    if left:
        return Action.LEFTFIRE if fire else Action.LEFT
    if right:
        return Action.RIGHTFIRE if fire else Action.RIGHT
    if fire:
        return Action.FIRE
    return Action.NOOP


def play(scale: int) -> None:
    import pygame

    env = JaxIceHockey()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    render_fn = jax.jit(env.render)

    _obs, state = reset_fn(jrandom.PRNGKey(0))
    h, w = env.consts.HEIGHT, env.consts.WIDTH

    pygame.init()
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("jax_icehockey render test — arrows move, space fire, R reset")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    _obs, state = reset_fn(jrandom.PRNGKey(0))

        action = get_action(pygame, pygame.key.get_pressed())
        step_out = step_fn(state, action)
        state = step_out[1]  # (obs, state, reward, done, info)

        img = to_uint8_image(render_fn(state))
        # pygame surfaces are (W, H), so transpose axes 0 and 1.
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--play", action="store_true", help="open an interactive pygame window")
    parser.add_argument("--out", default="icehockey_frame.png", help="snapshot output path")
    parser.add_argument("--scale", type=int, default=UPSCALE, help="upscale factor")
    parser.add_argument("--cpu", action="store_true", help="force JAX onto CPU")
    args = parser.parse_args()

    if args.play:
        play(args.scale)
    else:
        snapshot(args.out, args.scale)


if __name__ == "__main__":
    main()
