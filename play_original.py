"""Play the original ALE Ice Hockey with the keyboard, for side-by-side comparison
with the JAXAtari reimplementation.

Controls:
    Arrow keys     move (diagonals supported)
    Space          FIRE (shoot / body-check)
    Space + arrow  directional shot / check
    Esc            quit

Examples:
    python play_original.py
    python play_original.py --zoom 4 --fps 30 --difficulty 0
    python play_original.py --difficulty 3   # hardest CPU opponent
"""

import argparse

import ale_py
import gymnasium as gym
import pygame
from gymnasium.utils.play import play

gym.register_envs(ale_py)

# full_action_space=True so action indices are the standard 18-action ALE enum,
# matching JaxIceHockey.ACTION_SET (NOOP=0 ... DOWNLEFTFIRE=17).
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

# play() sorts each tuple internally, so the key order within a tuple does not matter.
KEY_BINDINGS = {
    (pygame.K_SPACE,): FIRE,
    (pygame.K_UP,): UP,
    (pygame.K_RIGHT,): RIGHT,
    (pygame.K_LEFT,): LEFT,
    (pygame.K_DOWN,): DOWN,
    (pygame.K_UP, pygame.K_RIGHT): UPRIGHT,
    (pygame.K_UP, pygame.K_LEFT): UPLEFT,
    (pygame.K_DOWN, pygame.K_RIGHT): DOWNRIGHT,
    (pygame.K_DOWN, pygame.K_LEFT): DOWNLEFT,
    (pygame.K_UP, pygame.K_SPACE): UPFIRE,
    (pygame.K_RIGHT, pygame.K_SPACE): RIGHTFIRE,
    (pygame.K_LEFT, pygame.K_SPACE): LEFTFIRE,
    (pygame.K_DOWN, pygame.K_SPACE): DOWNFIRE,
    (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): UPRIGHTFIRE,
    (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): UPLEFTFIRE,
    (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): DOWNRIGHTFIRE,
    (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): DOWNLEFTFIRE,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zoom", type=float, default=4.0, help="window zoom factor (default: 4)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default: 30)")
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="CPU opponent difficulty; 0 = simplest (default), 3 = hardest",
    )
    args = parser.parse_args()

    env = gym.make(
        "ALE/IceHockey-v5",
        render_mode="rgb_array",
        full_action_space=True,
        difficulty=args.difficulty,
    )
    play(env, keys_to_action=KEY_BINDINGS, noop=NOOP, zoom=args.zoom, fps=args.fps)


if __name__ == "__main__":
    main()
