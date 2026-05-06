import os
import sys

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import numpy as np
import imageio
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxatari.core import make as jaxatari_make
from jaxatari.environment import JAXAtariAction

DEFAULT_GIF_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_static", "gifs")
DISPLAY_SCALE = 4  # pygame window upscale (independent of GIF scale)


def _get_action_from_keys(env):
    """Map currently pressed pygame keys to an action index for this env."""
    import pygame
    keys = pygame.key.get_pressed()
    up    = keys[pygame.K_UP]    or keys[pygame.K_w]
    down  = keys[pygame.K_DOWN]  or keys[pygame.K_s]
    left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire  = keys[pygame.K_SPACE] or keys[pygame.K_RETURN]

    if   up and right and fire: name = "UPRIGHTFIRE"
    elif up and left  and fire: name = "UPLEFTFIRE"
    elif down and right and fire: name = "DOWNRIGHTFIRE"
    elif down and left  and fire: name = "DOWNLEFTFIRE"
    elif up   and fire: name = "UPFIRE"
    elif down and fire: name = "DOWNFIRE"
    elif left and fire: name = "LEFTFIRE"
    elif right and fire: name = "RIGHTFIRE"
    elif up    and right: name = "UPRIGHT"
    elif up    and left:  name = "UPLEFT"
    elif down  and right: name = "DOWNRIGHT"
    elif down  and left:  name = "DOWNLEFT"
    elif fire:  name = "FIRE"
    elif up:    name = "UP"
    elif down:  name = "DOWN"
    elif left:  name = "LEFT"
    elif right: name = "RIGHT"
    else:       name = "NOOP"

    const = getattr(JAXAtariAction, name, JAXAtariAction.NOOP)
    if hasattr(env, "ACTION_SET"):
        action_set = np.array(env.ACTION_SET)
        matches = np.where(action_set == int(const))[0]
        if len(matches):
            return jnp.array(int(matches[0]), dtype=jnp.int32)
    return jnp.array(0, dtype=jnp.int32)


def main():
    parser = argparse.ArgumentParser(description="Generate a GIF from a JAXAtari environment.")
    parser.add_argument("-g", "--game",   type=str, required=True, help="Game name (e.g. 'seaquest')")
    parser.add_argument("--frames",  type=int, default=300, help="Number of frames to capture")
    parser.add_argument("--fps",     type=int, default=15,  help="GIF frame rate")
    parser.add_argument("--scale",   type=int, default=2,   help="Pixel upscale factor for the GIF")
    parser.add_argument("--warmup",  type=int, default=0,   help="Frames to step through before recording starts")
    parser.add_argument("--play",    action="store_true",   help="Play manually instead of random actions")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  type=str, default=None, help="Output path (default: docs/_static/gifs/<game>.gif)")
    parser.add_argument("--cpu",     action="store_true")
    args = parser.parse_args()

    if args.output is None:
        os.makedirs(DEFAULT_GIF_DIR, exist_ok=True)
        args.output = os.path.join(DEFAULT_GIF_DIR, f"{args.game}.gif")

    env = jaxatari_make(args.game)
    jitted_reset  = jax.jit(env.reset)
    jitted_step   = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    key = jrandom.PRNGKey(args.seed)
    key, reset_key = jrandom.split(key)
    obs, state = jitted_reset(reset_key)

    action_space = env.action_space()
    action_key   = jrandom.PRNGKey(args.seed + 1)

    def random_action():
        nonlocal action_key
        action = action_space.sample(action_key)
        action_key, _ = jrandom.split(action_key)
        return action

    def step(state, action):
        nonlocal key
        obs, state, reward, done, info = jitted_step(state, action)
        if bool(done):
            key, reset_key = jrandom.split(key)
            obs, state = jitted_reset(reset_key)
        return state

    def capture(state):
        frame = np.array(jitted_render(state), dtype=np.uint8)
        if args.scale > 1:
            frame = frame.repeat(args.scale, axis=0).repeat(args.scale, axis=1)
        return frame

    # --- warmup (random in both modes) ---
    if args.warmup > 0:
        print(f"Warming up for {args.warmup} frames...")
        for _ in range(args.warmup):
            state = step(state, random_action())

    # --- record ---
    if args.play:
        import pygame
        pygame.init()
        sample_frame = np.array(jitted_render(state))
        h, w = sample_frame.shape[:2]
        window = pygame.display.set_mode((w * DISPLAY_SCALE, h * DISPLAY_SCALE))
        clock  = pygame.time.Clock()

        frames = []
        print(f"Recording {args.frames} frames — play now! ESC or Q to stop early.")

        running = True
        while running and len(frames) < args.frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            action = _get_action_from_keys(env)
            state  = step(state, action)
            frame  = capture(state)
            frames.append(frame)

            # display (always full DISPLAY_SCALE regardless of gif scale)
            display_frame = np.array(jitted_render(state), dtype=np.uint8)
            surf = pygame.surfarray.make_surface(np.transpose(display_frame, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (w * DISPLAY_SCALE, h * DISPLAY_SCALE))
            window.blit(surf, (0, 0))
            pygame.display.set_caption(f"Recording {len(frames)}/{args.frames}")
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
    else:
        frames = []
        print(f"Rendering {args.frames} frames for '{args.game}'...")
        for _ in range(args.frames):
            frames.append(capture(state))
            state = step(state, random_action())

    print(f"Saving {len(frames)} frames to {args.output} ...")
    imageio.mimsave(args.output, frames, fps=args.fps, loop=0)
    print("Done.")


if __name__ == "__main__":
    main()
