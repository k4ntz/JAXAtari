import os
import sys

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import numpy as np
import jax
import jax.random as jrandom
from PIL import Image

from jaxatari.core import make as jaxatari_make

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_static", "svgs")


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


def capture_game(game: str, warmup: int, seed: int, scale: int, pdf: bool, output: str | None) -> None:
    ext = "pdf" if pdf else "svg"

    if output is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        output = os.path.join(DEFAULT_OUT_DIR, f"{game}.{ext}")

    print(f"[{game}] Loading...")
    env = jaxatari_make(game)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    key = jrandom.PRNGKey(seed)
    key, reset_key = jrandom.split(key)
    obs, state = jitted_reset(reset_key)

    if warmup > 0:
        print(f"[{game}] Warming up for {warmup} frames...")
        action_space = env.action_space()
        action_key = jrandom.PRNGKey(seed + 1)
        for _ in range(warmup):
            action = action_space.sample(action_key)
            action_key, _ = jrandom.split(action_key)
            obs, state, reward, done, info = jitted_step(state, action)
            if bool(done):
                key, reset_key = jrandom.split(key)
                obs, state = jitted_reset(reset_key)

    frame = np.array(jitted_render(state), dtype=np.uint8)

    if pdf:
        img = Image.fromarray(frame)
        if scale > 1:
            img = img.resize(
                (frame.shape[1] * scale, frame.shape[0] * scale),
                resample=Image.NEAREST,
            )
        img.save(output, "PDF", resolution=72)
        print(f"[{game}] Saved PDF ({img.width}x{img.height} px, scale={scale}) → {output}")
    else:
        svg = frame_to_svg(frame, scale=scale)
        with open(output, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"[{game}] Saved SVG ({frame.shape[1]}x{frame.shape[0]} px, scale={scale}) → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Render an SVG (or PDF) screenshot of one or more JAXAtari games."
    )
    game_group = parser.add_mutually_exclusive_group(required=True)
    game_group.add_argument("-g", "--game", type=str, help="Single game name (e.g. 'seaquest')")
    game_group.add_argument(
        "--list", type=str, metavar="FILE",
        help="Text file with one game name per line.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Random frames to step before capturing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=1, help="Pixel size in output units (default: 1)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for single-game mode (default: docs/_static/svgs/<game>.<ext>). Ignored with --list.",
    )
    parser.add_argument("--pdf", action="store_true", help="Output PDF instead of SVG.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.list:
        with open(args.list, encoding="utf-8") as f:
            games = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if not games:
            print("No games found in list file.")
            sys.exit(1)
        print(f"Processing {len(games)} game(s) from {args.list}...")
        errors = []
        for game in games:
            try:
                capture_game(game, args.warmup, args.seed, args.scale, args.pdf, output=None)
            except Exception as e:
                print(f"[{game}] ERROR: {e}")
                errors.append(game)
        if errors:
            print(f"\nFailed for: {', '.join(errors)}")
            sys.exit(1)
    else:
        capture_game(args.game, args.warmup, args.seed, args.scale, args.pdf, args.output)


if __name__ == "__main__":
    main()
