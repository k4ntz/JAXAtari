import os
import sys

from jaxatari.environment import JAXAtariAction

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import numpy as np
import jax
import jax.random as jrandom
from PIL import Image
import gymnasium as gym
import ale_py  # noqa: F401  # Registers ALE environments for gymnasium.

from jaxatari.core import make as jaxatari_make
from jaxatari.core import list_available_games as jaxatari_list_games

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_static", "svgs")

# Mapping from JAXAtari env names to ALE game names (without ALE/ prefix and -v5 suffix).
JAXATARI_TO_ALE_GAMES = {
    # important
    "asteroids": "Asteroids",
    "bankheist": "BankHeist",
    "beamrider": "BeamRider",
    "breakout": "Breakout",
    "enduro": "Enduro",
    "freeway": "Freeway",
    "frostbite": "Frostbite",
    "gravitar": "Gravitar",
    "kangaroo": "Kangaroo",
    "montezumarevenge": "MontezumaRevenge",
    "phoenix": "Phoenix",
    "pong": "Pong",
    "qbert": "Qbert",
    "seaquest": "Seaquest",
    "skiing": "Skiing",
    "tennis": "Tennis",
    "timepilot": "TimePilot",
    "venture": "Venture",
    # remaining games
    "airraid": "AirRaid",
    "alien": "Alien",
    "amidar": "Amidar",
    "asterix": "Asterix",
    "atlantis": "Atlantis",
    "berzerk": "Berzerk",
    "blackjack": "Blackjack",
    "centipede": "Centipede",
    "choppercommand": "ChopperCommand",
    "fishingderby": "FishingDerby",
    "flagcapture": "FlagCapture",
    "galaxian": "Galaxian",
    "hangman": "Hangman",
    "hauntedhouse": "HauntedHouse",
    "humancannonball": "HumanCannonball",
    "kingkong": "KingKong",
    "klax": "Klax",
    "lasergates": "LaserGates",
    "mspacman": "MsPacman",
    "namethisgame": "NameThisGame",
    "pacman": "Pacman",
    "riverraid": "Riverraid",
    "sirlancelot": "SirLancelot",
    "spaceinvaders": "SpaceInvaders",
    "spacewar": "SpaceWar",
    "tetris": "Tetris",
    "tron": "Trondead",
    "turmoil": "Turmoil",
    "videocheckers": "VideoCheckers",
    "videocube": "VideoCube",
    "videopinball": "VideoPinball",
    "wordzapper": "WordZapper",
}


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


def parse_game_list_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def parse_csv_games(csv: str) -> list[str]:
    return [game.strip() for game in csv.split(",") if game.strip()]


def get_game_frame(game: str, warmup: int, seed: int) -> np.ndarray:
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
        noop = np.int32(JAXAtariAction.NOOP)
        for _ in range(warmup):
            obs, state, reward, done, info = jitted_step(state, noop)
            if bool(done):
                key, reset_key = jrandom.split(key)
                obs, state = jitted_reset(reset_key)

    return np.array(jitted_render(state), dtype=np.uint8)


def get_ale_frame(ale_game: str, warmup: int, seed: int) -> np.ndarray:
    env_id = f"ALE/{ale_game}-v5"
    print(f"[ALE:{ale_game}] Loading ({env_id})...")
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.0,
    )
    try:
        obs, info = env.reset(seed=seed)
        if warmup > 0:
            print(f"[ALE:{ale_game}] Warming up for {warmup} frames...")
            noop = int(JAXAtariAction.NOOP)
            for _ in range(warmup):
                obs, reward, terminated, truncated, info = env.step(noop)
                if terminated or truncated:
                    obs, info = env.reset()
        frame = env.render()
        return np.array(frame, dtype=np.uint8)
    finally:
        env.close()


def ale_env_exists(ale_game: str) -> bool:
    return f"ALE/{ale_game}-v5" in gym.registry


def create_mosaic(frames: list[np.ndarray], labels: list[str], columns: int, scale: int = 1, pad: int = 4) -> np.ndarray:
    if not frames:
        raise ValueError("No frames provided for mosaic.")

    heights = [frame.shape[0] for frame in frames]
    widths = [frame.shape[1] for frame in frames]
    max_h = max(heights)
    max_w = max(widths)
    rows = (len(frames) + columns - 1) // columns

    tile_h = max_h
    tile_w = max_w
    canvas_h = rows * tile_h + (rows + 1) * pad
    canvas_w = columns * tile_w + (columns + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        y0 = pad + row * (tile_h + pad)
        x0 = pad + col * (tile_w + pad)
        fh, fw = frame.shape[:2]

        # Center each frame within its tile for easier visual comparison.
        y_off = y0 + (tile_h - fh) // 2
        x_off = x0 + (tile_w - fw) // 2
        canvas[y_off:y_off + fh, x_off:x_off + fw] = frame

    if scale > 1:
        img = Image.fromarray(canvas)
        img = img.resize((canvas.shape[1] * scale, canvas.shape[0] * scale), resample=Image.NEAREST)
        canvas = np.array(img, dtype=np.uint8)

    return canvas


def create_pair_frame(left: np.ndarray, right: np.ndarray, pad: int = 4) -> np.ndarray:
    left_h, left_w = left.shape[:2]
    right_h, right_w = right.shape[:2]
    out_h = max(left_h, right_h) + 2 * pad
    out_w = left_w + right_w + 3 * pad
    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    left_y = pad + (max(left_h, right_h) - left_h) // 2
    right_y = pad + (max(left_h, right_h) - right_h) // 2
    left_x = pad
    right_x = left_w + 2 * pad

    canvas[left_y:left_y + left_h, left_x:left_x + left_w] = left
    canvas[right_y:right_y + right_h, right_x:right_x + right_w] = right
    return canvas


def save_frame(frame: np.ndarray, output: str, scale: int, pdf: bool, game: str | None = None) -> None:
    label = game if game is not None else "mosaic"
    if pdf:
        img = Image.fromarray(frame)
        if scale > 1:
            img = img.resize(
                (frame.shape[1] * scale, frame.shape[0] * scale),
                resample=Image.NEAREST,
            )
        img.save(output, "PDF", resolution=72)
        print(f"[{label}] Saved PDF ({img.width}x{img.height} px, scale={scale}) → {output}")
    else:
        svg = frame_to_svg(frame, scale=scale)
        with open(output, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"[{label}] Saved SVG ({frame.shape[1]}x{frame.shape[0]} px, scale={scale}) → {output}")


def capture_game(game: str, warmup: int, seed: int, scale: int, pdf: bool, output: str | None) -> None:
    ext = "pdf" if pdf else "svg"

    if output is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        output = os.path.join(DEFAULT_OUT_DIR, f"{game}.{ext}")

    frame = get_game_frame(game, warmup, seed)
    save_frame(frame, output, scale, pdf, game=game)


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
    game_group.add_argument(
        "--games", type=str, metavar="CSV",
        help="Comma-separated game names (e.g. 'seaquest,beamrider,phoenix').",
    )
    game_group.add_argument(
        "--all-jax-ale-pairs",
        action="store_true",
        help="Generate one ALE-vs-JAXAtari pair image for every game in the global mapping.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="NOOP steps before capture (JAXAtari and ALE)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=1, help="Pixel size in output units (default: 1)")
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output path for single-game mode (default: docs/_static/svgs/<game>.<ext>). "
            "With --all-jax-ale-pairs, this is treated as the output directory for pair files. "
            "Ignored with --list/--games."
        ),
    )
    parser.add_argument("--pdf", action="store_true", help="Output PDF instead of SVG.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--mosaic", action="store_true",
        help="With --list/--games, additionally save a combined comparison mosaic.",
    )
    parser.add_argument(
        "--mosaic-output", type=str, default=None,
        help="Path for mosaic output (default: docs/_static/svgs/mosaic_<games>.<ext>).",
    )
    parser.add_argument(
        "--columns", type=int, default=3,
        help="Column count for mosaic layout (default: 3).",
    )
    parser.add_argument(
        "--mosaic-pad", type=int, default=4,
        help="Padding (in source pixels) between images in the mosaic (default: 4).",
    )
    parser.add_argument(
        "--mosaic-only", action="store_true",
        help="With --list/--games, skip per-game files and only save the mosaic.",
    )
    parser.add_argument(
        "--pair-steps", type=int, default=20,
        help="NOOP step count before capture in --all-jax-ale-pairs mode (default: 20).",
    )
    args = parser.parse_args()

    if args.columns <= 0:
        print("--columns must be >= 1")
        sys.exit(1)
    if args.mosaic_pad < 0:
        print("--mosaic-pad must be >= 0")
        sys.exit(1)
    if args.pair_steps < 0:
        print("--pair-steps must be >= 0")
        sys.exit(1)

    if args.all_jax_ale_pairs:
        ext = "pdf" if args.pdf else "svg"
        pair_out_dir = args.output if args.output else DEFAULT_OUT_DIR
        os.makedirs(pair_out_dir, exist_ok=True)
        errors = []
        mapped_games = sorted(JAXATARI_TO_ALE_GAMES.items())
        available = set(jaxatari_list_games())
        print(f"Generating ALE/JAXAtari pair images for {len(mapped_games)} mapped games...")
        for jax_game, ale_game in mapped_games:
            if jax_game not in available:
                print(f"[{jax_game}] ERROR: game not registered in jaxatari.core")
                errors.append(jax_game)
                continue
            if not ale_env_exists(ale_game):
                print(f"[{jax_game}] SKIP: ALE/{ale_game}-v5 is not available in this installation")
                continue
            try:
                ale_frame = get_ale_frame(ale_game, args.pair_steps, args.seed)
                jax_frame = get_game_frame(jax_game, args.pair_steps, args.seed)
                pair = create_pair_frame(left=ale_frame, right=jax_frame, pad=args.mosaic_pad)
                output = os.path.join(pair_out_dir, f"pair_{jax_game}_ale_left_jax_right.{ext}")
                save_frame(pair, output, args.scale, args.pdf, game=f"pair:{jax_game}")
            except Exception as e:
                print(f"[{jax_game}] ERROR: {e}")
                errors.append(jax_game)
        if errors:
            print(f"\nFailed for: {', '.join(errors)}")
            sys.exit(1)
        return

    if args.list or args.games:
        games = parse_game_list_file(args.list) if args.list else parse_csv_games(args.games)
        if not games:
            print("No games found.")
            sys.exit(1)
        source = args.list if args.list else "--games"
        print(f"Processing {len(games)} game(s) from {source}...")
        errors = []
        frames = []
        ok_games = []
        for game in games:
            try:
                frame = get_game_frame(game, args.warmup, args.seed)
                frames.append(frame)
                ok_games.append(game)
                if not args.mosaic_only:
                    ext = "pdf" if args.pdf else "svg"
                    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
                    output = os.path.join(DEFAULT_OUT_DIR, f"{game}.{ext}")
                    save_frame(frame, output, args.scale, args.pdf, game=game)
            except Exception as e:
                print(f"[{game}] ERROR: {e}")
                errors.append(game)

        if args.mosaic and frames:
            ext = "pdf" if args.pdf else "svg"
            if args.mosaic_output is None:
                os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
                suffix = "_".join(ok_games[:4])
                if len(ok_games) > 4:
                    suffix += "_etc"
                mosaic_output = os.path.join(DEFAULT_OUT_DIR, f"mosaic_{suffix}.{ext}")
            else:
                mosaic_output = args.mosaic_output

            mosaic_frame = create_mosaic(
                frames=frames,
                labels=ok_games,
                columns=args.columns,
                scale=1,
                pad=args.mosaic_pad,
            )
            save_frame(mosaic_frame, mosaic_output, args.scale, args.pdf)

        if errors:
            print(f"\nFailed for: {', '.join(errors)}")
            sys.exit(1)
    else:
        capture_game(args.game, args.warmup, args.seed, args.scale, args.pdf, args.output)


if __name__ == "__main__":
    main()

