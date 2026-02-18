import sys
import ale_py
import gymnasium as gym
from gymnasium.utils.play import play
gym.register_envs(ale_py)

#!/usr/bin/env python3
"""
Minimal script to play Atari BattleZone with keyboard.

Requirements:
    pip install "gymnasium[atari]" pygame

Run:
    python bz.py
"""


def main():
        env_id = "ALE/BattleZone-v5"
        # Use rgb_array so gym's play utility displays frames via pygame
        env = gym.make(env_id, render_mode="rgb_array")
        try:
                # zoom increases on-screen size; adjust as needed
                play(env, zoom=4)
        except KeyboardInterrupt:
                pass
        finally:
                env.close()


if __name__ == "__main__":
        main()