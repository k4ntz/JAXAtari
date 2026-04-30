import os
import sys

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import numpy as np
import jax
import jax.random as jrandom

from jaxatari.core import make as jaxatari_make

DEFAULT_SVG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_static", "svgs")


def frame_to_svg(frame: np.ndarray, scale: int = 1) -> str:
    """Convert an (H, W, 3) uint8 frame to SVG using row run-length encoding."""
    h, w = frame.shape[:2]
    rects = []
    for y in range(h):
        x = 0
        while x < w:
            r, g, b = frame[y, x]
            run = 1
            while x + run < w and np.array_equal(frame[y, x + run], frame[y, x]):
                run += 1
            color = f"#{r:02x}{g:02x}{b:02x}"
            rects.append(
                f'<rect x="{x * scale}" y="{y * scale}"'
                f' width="{run * scale}" height="{scale}" fill="{color}"/>'
            )
            x += run

    body = "\n".join(rects)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{w * scale}" height="{h * scale}"'
        f' shape-rendering="crispEdges">\n{body}\n</svg>'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render a single SVG screenshot of a JAXAtari game."
    )
    parser.add_argument("-g", "--game", type=str, required=True, help="Game name (e.g. 'seaquest')")
    parser.add_argument("--warmup", type=int, default=0, help="Random frames to step before capturing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=1, help="Pixel size in SVG units (default: 1)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: docs/_static/svgs/<game>.svg)",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        os.makedirs(DEFAULT_SVG_DIR, exist_ok=True)
        args.output = os.path.join(DEFAULT_SVG_DIR, f"{args.game}.svg")

    env = jaxatari_make(args.game)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    key = jrandom.PRNGKey(args.seed)
    key, reset_key = jrandom.split(key)
    obs, state = jitted_reset(reset_key)

    if args.warmup > 0:
        print(f"Warming up for {args.warmup} frames...")
        action_space = env.action_space()
        action_key = jrandom.PRNGKey(args.seed + 1)
        for _ in range(args.warmup):
            action = action_space.sample(action_key)
            action_key, _ = jrandom.split(action_key)
            obs, state, reward, done, info = jitted_step(state, action)
            if bool(done):
                key, reset_key = jrandom.split(key)
                obs, state = jitted_reset(reset_key)

    frame = np.array(jitted_render(state), dtype=np.uint8)
    svg = frame_to_svg(frame, scale=args.scale)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved SVG ({frame.shape[1]}x{frame.shape[0]} px, scale={args.scale}) → {args.output}")


if __name__ == "__main__":
    main()
