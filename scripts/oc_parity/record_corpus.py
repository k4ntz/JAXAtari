#!/usr/bin/env python3
"""Record one OCAtari trajectory per implemented translator into the corpus.

Default is ``--mode random`` so the eval corpus can be built headlessly.
Replace with human ``--mode play`` recordings using the same output names when
you want intentional mid-game coverage.

Example:
  pyenv activate cuda_jax
  python scripts/oc_parity/record_corpus.py \\
    --corpus data/oc_parity_corpus --mode random --skip-frames 500 --steps 800
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record OC trajs for all translators")
    parser.add_argument("--corpus", type=str, default="data/oc_parity_corpus")
    parser.add_argument("--mode", choices=("random", "play"), default="random")
    parser.add_argument("--skip-frames", type=int, default=500)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="Optional jax keys (default: all implemented)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing *_play01.npz",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="play01",
        help="Filename tag: <jaxkey>_<tag>.npz (default play01)",
    )
    args = parser.parse_args()

    from oc_parity.translators.registry import (
        JAX_TO_OC_GAME,
        list_implemented_translators,
        print_skipped_games_note,
    )

    raw_dir = os.path.join(os.path.abspath(args.corpus), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    record_py = os.path.join(_THIS_DIR, "record_ocatari.py")
    games = args.games or list_implemented_translators()
    print_skipped_games_note()
    print(
        f"Recording corpus: mode={args.mode} skip={args.skip_frames} "
        f"steps={args.steps} games={len(games)} -> {raw_dir}"
    )

    failures: List[str] = []
    for jax_key in games:
        oc_name = JAX_TO_OC_GAME[jax_key]
        out = os.path.join(raw_dir, f"{jax_key}_{args.tag}.npz")
        if os.path.isfile(out) and not args.force:
            print(f"[skip] {jax_key}: exists {out}")
            continue
        cmd = [
            sys.executable,
            record_py,
            "-g",
            oc_name,
            "--mode",
            args.mode,
            "--skip-frames",
            str(args.skip_frames),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--out",
            out,
        ]
        print(f"\n=== {jax_key} / {oc_name} ===")
        print(" ", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=os.path.dirname(_SCRIPTS_DIR))
        if proc.returncode != 0:
            print(f"[fail] {jax_key} exit={proc.returncode}")
            failures.append(jax_key)
        else:
            print(f"[ok] {out}")

    if failures:
        raise SystemExit(f"Failed games: {', '.join(failures)}")
    print(f"\nDone. Trajectories in {raw_dir}")


if __name__ == "__main__":
    main()
